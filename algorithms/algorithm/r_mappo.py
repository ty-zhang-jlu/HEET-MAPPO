"""
# @Time    : 2021/7/1 6:52 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_mappo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from global_mappo.utils.util import get_gard_norm, huber_loss, mse_loss
from global_mappo.utils.valuenorm import ValueNorm
from global_mappo.algorithms.utils.util import check

class DynamicCoeffNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicCoeffNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 输出范围在[0, 1]
        return x

class RMAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.C_sigema = args._C_sigema
        self.N_beta = args.num_agents
        self.action_dim = args.action_dims
        self.kesai = args._kesai
        self.tau = args.gamma
        self.iter_time = args.ppo_epoch
        self.lr = args.lr


        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            self.value_normalizer = self.policy.critic.q_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def gradient_penalty(self, actor_network, states, rnn_states, masks, actions, lambda_gp):
        """
        计算策略网络的梯度惩罚项。
        :param actor_network: 策略网络（actor）。
        :param states: 输入状态（NumPy 数组）。
        :param rnn_states: RNN 隐藏状态（NumPy 数组）。
        :param masks: 掩码（NumPy 数组）。
        :param actions: 动作（NumPy 数组）。
        :param lambda_gp: 梯度惩罚的权重系数。
        :return: 梯度惩罚项。
        """


        # 将数据转换为 PyTorch 张量
        states = torch.tensor(states, dtype=torch.float32, device=self.device, requires_grad=True)
        rnn_states = torch.tensor(rnn_states, dtype=torch.float32, device=self.device)
        masks = torch.tensor(masks, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        # 检查输入数据是否包含 NaN 或 Inf
        if torch.isnan(states).any() or torch.isinf(states).any():
            print("Warning: NaN or Inf values found in states!")
            # states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(rnn_states).any() or torch.isinf(rnn_states).any():
            print("Warning: NaN or Inf values found in rnn_states!")
            # rnn_states = torch.nan_to_num(rnn_states, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(masks).any() or torch.isinf(masks).any():
            print("Warning: NaN or Inf values found in masks!")
            # masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print("Warning: NaN or Inf values found in actions!")
            # actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)

        # 禁用 CuDNN 后端
        with torch.backends.cudnn.flags(enabled=False):
            # 计算动作概率
            _, action_log_probs, _ = actor_network(states, rnn_states, masks)  # 只解包 action_log_probs

            # 使用 Normal 分布计算对数概率
            dist = torch.distributions.Normal(loc=action_log_probs, scale=1.0)  # 假设标准差为 1.0
            log_probs = dist.log_prob(actions).sum(dim=-1)  # 对多维度动作求和

            # 计算梯度
            gradients = torch.autograd.grad(
                outputs=log_probs,
                inputs=states,
                grad_outputs=torch.ones_like(log_probs),
                create_graph=True,
                retain_graph=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)  # 展平梯度
            gradient_norm = gradients.norm(2, dim=1)  # 计算 L2 范数

            # 计算梯度惩罚项
            gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
            # 检查梯度是否包含 NaN 或 Inf
            if torch.isnan(gradients).any() or torch.isinf(gradients).any():
                print("Warning: NaN or Inf values found in gradients!")

            return gradient_penalty

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def compute_cei(self, joint_action_log_probs, old_action_log_probs_batch_1, old_action_log_probs_batch_2, old_action_log_probs_batch_3, share_obs_batch, obs, rnn_states_actor,
                    rnn_states_actor_BS, rnn_states_actor_MBS, rnn_states_critic_batch, new_action_1, new_action_2, new_action_3, masks, available_actions, active_masks,
                    dist_entropy_1, dist_entropy_2, dist_entropy_3, grad_1, grad_2, grad_3, values, q_values, values_BS, q_values_BS, values_MBS, q_values_MBS,
                    values_joint, q_values_joint):
        # 1. 原始动作分布
        old_joint_action = torch.cat([old_action_log_probs_batch_1, old_action_log_probs_batch_2, old_action_log_probs_batch_3], dim=1)
        # 2. 原始熵和KL
        entropy_1 = - (torch.from_numpy(new_action_1).to(device='cuda:0') * torch.log(torch.from_numpy(new_action_1).to(device='cuda:0'))).sum(dim=-1)
        entropy_1 = torch.nan_to_num(entropy_1, nan=1e-5)
        entropy_2 = - (torch.from_numpy(new_action_2).to(device='cuda:0') * torch.log(
            torch.from_numpy(new_action_2).to(device='cuda:0'))).sum(dim=-1)
        entropy_2 = torch.nan_to_num(entropy_2, nan=1e-5)
        entropy_3 = - (torch.from_numpy(new_action_3).to(device='cuda:0') * torch.log(
            torch.from_numpy(new_action_3).to(device='cuda:0'))).sum(dim=-1)
        entropy_3 = torch.nan_to_num(entropy_3, nan=1e-5)
        # entropy_1 = dist_entropy_1
        # entropy_2 = dist_entropy_2
        # entropy_3 = dist_entropy_3
        joint_kl_old = F.kl_div(old_joint_action, joint_action_log_probs.exp(), reduction='batchmean')

        # 3. 扰动智能体i的熵
        grad_ent_1 = grad_1[1]
        new_action_1_perturbed = torch.from_numpy(new_action_1).to(device='cuda:0').clone()
        new_action_1_perturbed += 1e-5 * grad_ent_1
        new_action_1_perturbed = new_action_1_perturbed / new_action_1_perturbed.sum(dim=-1, keepdim=True)

        grad_ent_2 = grad_2[1]
        new_action_2_perturbed = torch.from_numpy(new_action_2).to(device='cuda:0').clone()
        new_action_2_perturbed += 1e-5 * grad_ent_2
        new_action_2_perturbed = new_action_2_perturbed / new_action_2_perturbed.sum(dim=-1, keepdim=True)

        grad_ent_3 = grad_3[1]
        new_action_3_perturbed = torch.from_numpy(new_action_3).to(device='cuda:0').clone()
        new_action_3_perturbed += 1e-5 * grad_ent_3
        new_action_3_perturbed = new_action_3_perturbed / new_action_3_perturbed.sum(dim=-1, keepdim=True)

        # 4. 计算扰动后的指标
        joint_actions_1 = torch.cat([new_action_1_perturbed, torch.from_numpy(new_action_2).to(device='cuda:0'), torch.from_numpy(new_action_3).to(device='cuda:0')], dim=1)
        joint_actions_2 = torch.cat([torch.from_numpy(new_action_1).to(device='cuda:0'), new_action_2_perturbed, torch.from_numpy(new_action_3).to(device='cuda:0')], dim=1)
        joint_actions_3 = torch.cat([torch.from_numpy(new_action_1).to(device='cuda:0'), torch.from_numpy(new_action_2).to(device='cuda:0'), new_action_3_perturbed], dim=1)
        joint_rnn_states = rnn_states_actor + rnn_states_actor_BS + rnn_states_actor_MBS
        joint_action_log_probs_perturbed_1, _ = self.policy.actor_joint.evaluate_actions_joint(obs,
                                                             joint_rnn_states,
                                                             joint_actions_1,
                                                             masks,
                                                             available_actions if available_actions is not None else None,
                                                             active_masks if active_masks is not None else None)
        joint_action_log_probs_perturbed_2, _ = self.policy.actor_joint.evaluate_actions_joint(obs,
                                                                                               joint_rnn_states,
                                                                                               joint_actions_2,
                                                                                               masks,
                                                                                               available_actions if available_actions is not None else None,
                                                                                               active_masks if active_masks is not None else None)
        joint_action_log_probs_perturbed_3, _ = self.policy.actor_joint.evaluate_actions_joint(obs,
                                                                                               joint_rnn_states,
                                                                                               joint_actions_3,
                                                                                               masks,
                                                                                               available_actions if available_actions is not None else None,
                                                                                               active_masks if active_masks is not None else None)
        # 计算价值函数和动作概率
        values_joint_new_1, q_values_joint_new_1, values_new_1, q_values_new_1, values_BS_new_1, q_values_BS_new_1, values_MBS_new_1, q_values_MBS_new_1,\
        _, dist_entropy_new_1, _, dist_entropy_2, _, dist_entropy_3, *_\
            = self.policy.evaluate_actions(share_obs_batch, obs, rnn_states_actor, rnn_states_actor_BS, rnn_states_actor_MBS,
                                           rnn_states_critic_batch,
                                           new_action_1_perturbed, new_action_2, new_action_3, masks, available_actions, active_masks)
        values_joint_new_2, q_values_joint_new_2, values_new_2, q_values_new_2, values_BS_new_2, q_values_BS_new_2, values_MBS_new_2, q_values_MBS_new_2,\
        _, dist_entropy_1, _, dist_entropy_new_2, _, dist_entropy_3, *_\
            = self.policy.evaluate_actions(share_obs_batch, obs, rnn_states_actor, rnn_states_actor_BS, rnn_states_actor_MBS,
                                           rnn_states_critic_batch,
                                           new_action_1, new_action_2_perturbed, new_action_3, masks, available_actions, active_masks)
        values_joint_new_3, q_values_joint_new_3, values_new_3, q_values_new_3, values_BS_new_3, q_values_BS_new_3, values_MBS_new_2, q_values_MBS_new_2, \
        _, dist_entropy_1, _, dist_entropy_2, _, dist_entropy_new_3, *_\
            = self.policy.evaluate_actions(share_obs_batch, obs, rnn_states_actor, rnn_states_actor_BS, rnn_states_actor_MBS,
                                           rnn_states_critic_batch,
                                           new_action_1, new_action_2, new_action_3_perturbed, masks, available_actions,
                                           active_masks)

        joint_kl_new_1 = F.kl_div(old_joint_action, joint_action_log_probs_perturbed_1.exp(), reduction='batchmean')
        joint_kl_new_2 = F.kl_div(old_joint_action, joint_action_log_probs_perturbed_2.exp(), reduction='batchmean')
        joint_kl_new_3 = F.kl_div(old_joint_action, joint_action_log_probs_perturbed_3.exp(), reduction='batchmean')
        delta_KL_1 = joint_kl_new_1 - joint_kl_old
        delta_KL_2 = joint_kl_new_2 - joint_kl_old
        delta_KL_3 = joint_kl_new_3 - joint_kl_old
        delta_q_value_1 = (q_values_joint_new_1 - q_values_joint).mean()
        delta_q_value_2 = (q_values_joint_new_2 - q_values_joint).mean()
        delta_q_value_3 = (q_values_joint_new_3 - q_values_joint).mean()

        entropies_perturbed_1 = - (new_action_1_perturbed * torch.log(new_action_1_perturbed)).sum(dim=-1)
        entropies_perturbed_1 = torch.nan_to_num(entropies_perturbed_1, nan=1e-5)
        delta_H_i_1 = (entropies_perturbed_1 - entropy_1).mean()
        CEI_1 = delta_KL_1 / (delta_H_i_1 + 1e-8)
        ERS_1 = delta_q_value_1 / (delta_H_i_1 + 1e-8)

        entropies_perturbed_2 = - (new_action_2_perturbed * torch.log(new_action_2_perturbed)).sum(dim=-1)
        entropies_perturbed_2 = torch.nan_to_num(entropies_perturbed_2, nan=1e-5)
        delta_H_i_2 = (entropies_perturbed_2 - entropy_2).mean()
        CEI_2 = delta_KL_2 / (delta_H_i_2 + 0.01)
        ERS_2 = delta_q_value_2 / (delta_H_i_2 + 0.01)

        entropies_perturbed_3 = - (new_action_3_perturbed * torch.log(new_action_3_perturbed)).sum(dim=-1)
        entropies_perturbed_3 = torch.nan_to_num(entropies_perturbed_3, nan=1e-5)
        delta_H_i_3 = (entropies_perturbed_3 - entropy_3).mean()
        CEI_3 = delta_KL_3 / (delta_H_i_3 + 0.01)
        ERS_3 = delta_q_value_3 / (delta_H_i_3 + 0.01)

        # 5. 返回CEI
        return CEI_1, CEI_2, CEI_3, ERS_1, ERS_2, ERS_3


    def ppo_update(self, sample, vary_L, vary_noise, beta_t, update_actor=True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_BS_batch, rnn_states_MBS_batch, rnn_states_critic_batch, actions_batch, actions_BS_batch, \
        actions_MBS_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, old_action_log_probs_BS_batch, \
        old_action_log_probs_MBS_batch, adv_targ, available_actions_batch = sample

        old_action_log_probs_batch_1 = old_action_log_probs_batch  # 前16个智能体的动作概率
        old_action_log_probs_batch_2 = old_action_log_probs_BS_batch  # 最后一个智能体（BS）的动作概率
        old_action_log_probs_batch_3 = old_action_log_probs_MBS_batch

        # 计算每个张量的最小值和最大值
        min_actions_batch = np.min(actions_batch)
        max_actions_batch = np.max(actions_batch)

        min_actions_BS_batch = np.min(actions_BS_batch)
        max_actions_BS_batch = np.max(actions_BS_batch)

        # 进行 Min-Max 归一化
        normalized_actions_batch = (actions_batch - min_actions_batch) / (max_actions_batch - min_actions_batch)
        normalized_actions_BS_batch = (actions_BS_batch - min_actions_BS_batch) / (
                    max_actions_BS_batch - min_actions_BS_batch)

        # 计算归一化后的上界（最大值）
        B_up_1 = np.max(normalized_actions_batch)
        B_up_2 = np.max(normalized_actions_BS_batch)
        # print("归一化动作值上界：", B_up_1, B_up_2)

        # 将数据转换为 PyTorch 张量
        old_action_log_probs_batch_1 = check(old_action_log_probs_batch_1).to(**self.tpdv)  # RIS的每个元件
        old_action_log_probs_batch_2 = check(old_action_log_probs_batch_2).to(**self.tpdv)  # BS的旧策略
        old_action_log_probs_batch_3 = check(old_action_log_probs_batch_3).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 计算价值函数和动作概率
        values_joint, q_values_joint, values, q_values, values_BS, q_values_BS, values_MBS, q_values_MBS, action_log_probs_1, dist_entropy_1,\
        action_log_probs_2, dist_entropy_2, action_log_probs_3, dist_entropy_3, joint_action_log_probs, joint_entropy, \
        grad_1, grad_2, grad_3\
            = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_BS_batch,
            rnn_states_MBS_batch,
            rnn_states_critic_batch,
            actions_batch,
            actions_BS_batch,
            actions_MBS_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch)
        joint_entropy = dist_entropy_1 + dist_entropy_2 + dist_entropy_3

        # 计算 Q 值误差（当前 Critic vs Target Critic 或贝尔曼目标）
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)  # 更新运行统计量
            target = self.value_normalizer.normalize(return_batch)  # 归一化目标
        else:
            target = return_batch
        q_error_1 = torch.sqrt(torch.mean(torch.square(values - target))) + vary_noise
        q_error_2 = torch.sqrt(torch.mean(torch.square(values_BS - target))) + vary_noise

        # # 计算 KL 散度
        kl_div = F.kl_div(old_action_log_probs_batch_1, action_log_probs_1.exp(), reduction='batchmean')  # RIS的策略之间的KL散度
        kl_div_2 = F.kl_div(old_action_log_probs_batch_2, action_log_probs_2.exp(), reduction='batchmean')  # BS的策略的KL散度值
        kl_div_3 = F.kl_div(old_action_log_probs_batch_3, action_log_probs_3.exp(), reduction='batchmean')
        # print(action_log_probs_1.shape, action_log_probs_2.shape)

        cei_1, cei_2, cei_3, ers_1, ers_2, ers_3= self.compute_cei(joint_action_log_probs, old_action_log_probs_batch_1, old_action_log_probs_batch_2, old_action_log_probs_batch_3,
                    share_obs_batch, obs_batch, rnn_states_batch, rnn_states_BS_batch, rnn_states_MBS_batch, rnn_states_critic_batch, actions_batch,
                                 actions_BS_batch, actions_MBS_batch, masks_batch, available_actions_batch, active_masks_batch, dist_entropy_1, dist_entropy_2, dist_entropy_3,
                                grad_1, grad_2, grad_3, values, q_values, values_BS, q_values_BS, values_MBS, q_values_MBS, values_joint, q_values_joint)
        # print("ERS指标值:", ers_1, ers_2)

        # 计算梯度惩罚
        gp_1 = self.gradient_penalty(self.policy.actor, obs_batch, rnn_states_batch, masks_batch, actions_batch,
                                     vary_L)
        # gp_2 = self.gradient_penalty(self.policy.actor_BS, obs_batch, rnn_states_BS_batch, masks_batch, actions_BS_batch,
        #                              vary_L)
        C_sigema_1 = self.C_sigema / dist_entropy_1
        C_sigema_2 = self.C_sigema / dist_entropy_2
        # print("调整C_sigema的实际大小：", C_sigema_1)
        lr_1 = (1 - self.tau) / (0.1 * dist_entropy_1 * self.N_beta * vary_L * torch.sqrt(0.1 * dist_entropy_1  * vary_L) * ( B_up_1 + q_error_1))
        lr_2 = (1 - self.tau) / (0.1 * dist_entropy_2 * self.N_beta * vary_L * torch.sqrt(0.1 * dist_entropy_2  * vary_L) * ( B_up_2 + q_error_2))
        lr_1_1 = (1 - self.tau) / (
                    C_sigema_1 * self.N_beta * vary_L * torch.sqrt(C_sigema_1 * vary_L) * (
                        B_up_1 + q_error_1))
        lr_2_1 = (1 - self.tau) / (
                    C_sigema_2 * self.N_beta * vary_L * torch.sqrt(C_sigema_2 * vary_L) * (
                        B_up_2 + q_error_2))
        # print("学习率：", lr_1, lr_2)
        entropy_coef_1 = (vary_L ** 2 * B_up_1 ** 2 + 2 * q_error_1) / np.sqrt(self.iter_time) \
                         / ((vary_L ** 3) * (lr_1_1 ** 2) + 2 * vary_L)
        entropy_coef_2 = (vary_L ** 2 * B_up_2 ** 2 + 2 * q_error_2) / np.sqrt(self.iter_time) \
                         / ((vary_L ** 3) * (lr_2_1 ** 2) + 2 * vary_L)
        # print("正则化系数：", entropy_coef_1, entropy_coef_2)

        # 计算重要性采样权重
        imp_weights_1 = torch.exp(action_log_probs_1 - old_action_log_probs_batch_1)
        imp_weights_2 = torch.exp(action_log_probs_2 - old_action_log_probs_batch_2)
        imp_weights_3 = torch.exp(action_log_probs_3 - old_action_log_probs_batch_3)

        # 计算策略损失
        surr1_1 = imp_weights_1 * adv_targ
        surr1_2 = torch.clamp(imp_weights_1, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        surr2_1 = imp_weights_2 * adv_targ
        surr2_2 = torch.clamp(imp_weights_2, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        surr3_1 = imp_weights_3 * adv_targ
        surr3_2 = torch.clamp(imp_weights_3, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1_1, surr1_2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            policy_action_loss_2 = (-torch.sum(torch.min(surr2_1, surr2_2),
                                               dim=-1,
                                               keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            policy_action_loss_3 = (-torch.sum(torch.min(surr3_1, surr3_2),
                                               dim=-1,
                                               keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()

        else:
            policy_action_loss = -torch.sum(torch.min(surr1_1, surr1_2), dim=-1, keepdim=True).mean()
            policy_action_loss_2 = -torch.sum(torch.min(surr2_1, surr2_2), dim=-1, keepdim=True).mean()
            policy_action_loss_3 = -torch.sum(torch.min(surr3_1, surr3_2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        policy_loss_2 = policy_action_loss_2
        policy_loss_3 = policy_action_loss_3

        # self.policy.actor_optimizer.param_groups[0]['lr'] = lr_1
        # self.policy.actor_optimizer_BS.param_groups[0]['lr'] = lr_2
        # 更新 actor 网络
        self.policy.actor_optimizer.zero_grad()
        self.policy.actor_optimizer_BS.zero_grad()
        self.policy.actor_optimizer_MBS.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy_1 * self.entropy_coef).backward(retain_graph=True)
            (policy_loss_2 - dist_entropy_2 * self.entropy_coef).backward(retain_graph=True)
            (policy_loss_3 - dist_entropy_3 * self.entropy_coef).backward(retain_graph=True)

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            actor_grad_norm_2 = nn.utils.clip_grad_norm_(self.policy.actor_BS.parameters(), self.max_grad_norm)
            actor_grad_norm_3 = nn.utils.clip_grad_norm_(self.policy.actor_MBS.parameters(), self.max_grad_norm)

        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            actor_grad_norm_2 = get_gard_norm(self.policy.actor_BS.parameters())
            actor_grad_norm_3 = get_gard_norm(self.policy.actor_MBS.parameters())

        self.policy.actor_optimizer.step()
        self.policy.actor_optimizer_BS.step()
        self.policy.actor_optimizer_MBS.step()

        value_loss_1 = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        value_loss_2 = self.cal_value_loss(values_BS, value_preds_batch, return_batch, active_masks_batch)
        value_loss_3 = self.cal_value_loss(values_MBS, value_preds_batch, return_batch, active_masks_batch)
        # 合并损失统一更新
        total_loss = (value_loss_1 + value_loss_2 + value_loss_3) * self.value_loss_coef

        # 统一反向传播
        self.policy.critic_optimizer_joint.zero_grad()
        total_loss.backward(retain_graph=True)
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic_joint.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic_joint.parameters())
        self.policy.critic_optimizer_joint.step()

        return value_loss_1, critic_grad_norm, policy_loss, policy_loss_2, policy_loss_3, dist_entropy_1, dist_entropy_2, dist_entropy_3, actor_grad_norm, actor_grad_norm_2,\
               actor_grad_norm_3, imp_weights_1, imp_weights_2, imp_weights_3, entropy_coef_1, entropy_coef_2, lr_1, lr_2, B_up_1, B_up_2, q_error_1, cei_1, cei_2, cei_3, joint_entropy,\
               ers_1, ers_2, ers_3

    def train_1(self, buffer, vary_L, vary_noise, beta_t, update_actor=True):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_loss_2'] = 0
        train_info['policy_loss_3'] = 0
        train_info['dist_entropy_1'] = 0
        train_info['dist_entropy_2'] = 0
        train_info['dist_entropy_3'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['actor_grad_norm_2'] = 0
        train_info['actor_grad_norm_3'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['ratio_2'] = 0
        train_info['ratio_3'] = 0
        train_info['kl'] = 0
        train_info['kl_2'] = 0
        train_info['lr_1'] = 0
        train_info['lr_2'] = 0
        train_info['Bup_1'] = 0
        train_info['Bup_2'] = 0
        train_info['Q_err'] = 0
        train_info['Cei_1'] = 0
        train_info['Cei_2'] = 0
        train_info['Cei_3'] = 0
        train_info['J_T'] = 0
        train_info['Ers_1'] = 0
        train_info['Ers_2'] = 0
        train_info['Ers_3'] = 0

        # 生成所有 mini-batch 的数据生成器
        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        else:
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        # 将所有 mini-batch 数据存储到列表中
        all_samples = list(data_generator)
        k_1 = []
        k_2 = []
        l_1 = []
        l_2 = []
        B_1 = []
        B_2 = []
        Q_1 = []
        C_1 = []
        C_2 = []
        C_3 = []
        J_e = []
        E_1 = []
        E_2 = []
        E_3 = []

        for _ in range(self.ppo_epoch):
            # 随机选择一个 mini-batch
            sample = all_samples[np.random.randint(0, len(all_samples))]
            value_loss, critic_grad_norm, policy_loss, policy_loss_2, policy_loss_3, dist_entropy_1, dist_entropy_2, dist_entropy_3, actor_grad_norm, actor_grad_norm_2, \
            actor_grad_norm_3, imp_weights_1, imp_weights_2, imp_weights_3, entropy_coef_1, entropy_coef_2, lr_1, lr_2, B_up_1, B_up_2, Q_error, cei_1, cei_2, cei_3, joint_entropy,\
            ers_1, ers_2, ers_3 = self.ppo_update(sample, vary_L, vary_noise, beta_t, update_actor)

            train_info['value_loss'] += value_loss.item()
            train_info['policy_loss'] += policy_loss.item()
            train_info['policy_loss_2'] += policy_loss_2.item()
            train_info['policy_loss_3'] += policy_loss_3.item()
            train_info['dist_entropy_1'] += dist_entropy_1.item()
            train_info['dist_entropy_2'] += dist_entropy_2.item()
            train_info['dist_entropy_3'] += dist_entropy_3.item()
            train_info['actor_grad_norm'] += actor_grad_norm
            train_info['actor_grad_norm_2'] += actor_grad_norm_2
            train_info['actor_grad_norm_3'] += actor_grad_norm_3
            train_info['critic_grad_norm'] += critic_grad_norm
            train_info['ratio'] += imp_weights_1.mean()
            train_info['ratio_2'] += imp_weights_2.mean()
            train_info['ratio_3'] += imp_weights_3.mean()
            k_1.append(entropy_coef_1)
            k_2.append(entropy_coef_2)
            l_1.append(lr_1)
            l_2.append(lr_2)
            B_1.append(B_up_1)
            B_2.append(B_up_2)
            Q_1.append(Q_error)
            C_1.append(cei_1)
            C_2.append(cei_2)
            C_3.append(cei_3)
            J_e.append(joint_entropy)
            E_1.append(ers_1)
            E_2.append(ers_2)
            E_3.append(ers_3)
        train_info['kl'] = torch.stack(k_1).mean().item()
        train_info['kl_2'] = torch.stack(k_2).mean().item()
        train_info['lr_1'] = torch.stack(l_1).mean().item()
        train_info['lr_2'] = torch.stack(l_2).mean().item()
        train_info['Bup_1'] = np.mean(B_1)
        train_info['Bup_2'] = np.mean(B_2)
        # train_info['Bup_1'] = torch.stack(B_1).mean().item()
        # train_info['Bup_2'] = torch.stack(B_2).mean().item()
        train_info['Q_err'] = torch.stack(Q_1).mean().item()
        train_info['Cei_1'] = torch.stack(C_1).mean().item()
        train_info['Cei_2'] = torch.stack(C_2).mean().item()
        train_info['Cei_3'] = torch.stack(C_3).mean().item()
        train_info['J_T'] = torch.stack(J_e).mean().item()
        train_info['Ers_1'] = torch.stack(E_1).mean().item()
        train_info['Ers_2'] = torch.stack(E_2).mean().item()
        train_info['Ers_3'] = torch.stack(E_3).mean().item()


        return train_info

    def train_2(self, buffer, vary_L, vary_noise, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_loss_2'] = 0
        train_info['policy_loss_3'] = 0
        train_info['dist_entropy_1'] = 0
        train_info['dist_entropy_2'] = 0
        train_info['dist_entropy_3'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['actor_grad_norm_2'] = 0
        train_info['actor_grad_norm_3'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['ratio_2'] = 0
        train_info['ratio_3'] = 0
        train_info['kl'] = 0
        train_info['kl_2'] = 0
        train_info['lr_1'] = 0
        train_info['lr_2'] = 0
        train_info['Bup_1'] = 0
        train_info['Bup_2'] = 0
        train_info['Q_err'] = 0
        train_info['Cei_1'] = 0
        train_info['Cei_2'] = 0
        train_info['Cei_3'] = 0
        train_info['J_T'] = 0
        train_info['Ers_1'] = 0
        train_info['Ers_2'] = 0
        train_info['Ers_3'] = 0

        k_1 = []
        k_2 = []
        l_1 = []
        l_2 = []
        B_1 = []
        B_2 = []
        Q_1 = []
        C_1 = []
        C_2 = []
        C_3 = []
        J_e = []
        E_1 = []
        E_2 = []
        E_3 = []

        for _ in range(self.ppo_epoch):
            # 生成所有 mini-batch 的数据生成器
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, policy_loss_2, policy_loss_3, dist_entropy_1, dist_entropy_2, dist_entropy_3, actor_grad_norm, actor_grad_norm_2, \
                actor_grad_norm_3, imp_weights_1, imp_weights_2, imp_weights_3, entropy_coef_1, entropy_coef_2, lr_1, lr_2, B_up_1, B_up_2, Q_error, cei_1, cei_2, cei_3, joint_entropy, \
                ers_1, ers_2, ers_3 = self.ppo_update(sample, vary_L, vary_noise, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_loss_2'] += policy_loss_2.item()
                train_info['policy_loss_3'] += policy_loss_3.item()
                train_info['dist_entropy_1'] += dist_entropy_1.item()
                train_info['dist_entropy_2'] += dist_entropy_2.item()
                train_info['dist_entropy_3'] += dist_entropy_3.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['actor_grad_norm_2'] += actor_grad_norm_2
                train_info['actor_grad_norm_3'] += actor_grad_norm_3
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights_1.mean()
                train_info['ratio_2'] += imp_weights_2.mean()
                train_info['ratio_3'] += imp_weights_3.mean()
                k_1.append(entropy_coef_1)
                k_2.append(entropy_coef_2)
                l_1.append(lr_1)
                l_2.append(lr_2)
                B_1.append(B_up_1)
                B_2.append(B_up_2)
                Q_1.append(Q_error)
                C_1.append(cei_1)
                C_2.append(cei_2)
                C_3.append(cei_3)
                J_e.append(joint_entropy)
                E_1.append(ers_1)
                E_2.append(ers_2)
                E_3.append(ers_3)
                train_info['kl'] = torch.stack(k_1).mean().item()
                train_info['kl_2'] = torch.stack(k_2).mean().item()
                train_info['lr_1'] = torch.stack(l_1).mean().item()
                train_info['lr_2'] = torch.stack(l_2).mean().item()
                train_info['Bup_1'] = np.mean(B_1)
                train_info['Bup_2'] = np.mean(B_2)
                # train_info['Bup_1'] = torch.stack(B_1).mean().item()
                # train_info['Bup_2'] = torch.stack(B_2).mean().item()
                train_info['Q_err'] = torch.stack(Q_1).mean().item()
                train_info['Cei_1'] = torch.stack(C_1).mean().item()
                train_info['Cei_2'] = torch.stack(C_2).mean().item()
                train_info['Cei_3'] = torch.stack(C_3).mean().item()
                train_info['J_T'] = torch.stack(J_e).mean().item()
                train_info['Ers_1'] = torch.stack(E_1).mean().item()
                train_info['Ers_2'] = torch.stack(E_2).mean().item()
                train_info['Ers_3'] = torch.stack(E_3).mean().item()
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.actor_BS.train()
        self.policy.actor_MBS.train()
        self.policy.actor_joint.train()
        self.policy.critic.train()
        self.policy.critic_BS.train()
        self.policy.critic_MBS.train()
        self.policy.critic_joint.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.actor_BS.eval()
        self.policy.actor_MBS.eval()
        self.policy.actor_joint.eval()
        self.policy.critic.eval()
        self.policy.critic_BS.eval()
        self.policy.critic_MBS.eval()
        self.policy.critic_joint.eval()

