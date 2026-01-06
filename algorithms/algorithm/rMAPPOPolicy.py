"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : rMAPPOPolicy.py
"""

import torch
from global_mappo.algorithms.algorithm.r_actor_critic import R_Actor, R_Critic, DynamicCoeffNetwork
from global_mappo.utils.util import update_linear_schedule
import time
import numpy as np


class RMAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space


        self.actor = R_Actor(args, self.obs_space, self.act_space[0], self.device)
        self.actor_BS = R_Actor(args, self.obs_space, self.act_space[1], self.device)
        self.actor_MBS = R_Actor(args, self.obs_space, self.act_space[2], self.device)
        self.actor_joint = R_Actor(args, self.obs_space, self.act_space[0] + self.act_space[1] + self.act_space[2], self.device)
        self.critic = R_Critic(args, self.obs_space, self.act_space[0], self.device)
        self.critic_BS = R_Critic(args, self.obs_space, self.act_space[1], self.device)
        self.critic_MBS = R_Critic(args, self.obs_space, self.act_space[2], self.device)
        self.critic_joint = R_Critic(args, self.obs_space, self.act_space[0] * 16 + self.act_space[1] + self.act_space[2], self.device)
        self.kl_dynamic_coeff_network = DynamicCoeffNetwork(self.obs_space[0], hidden_dim=64)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.actor_optimizer_BS = torch.optim.Adam(self.actor_BS.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.actor_optimizer_MBS = torch.optim.Adam(self.actor_MBS.parameters(),
                                                   lr=self.lr, eps=self.opti_eps,
                                                   weight_decay=self.weight_decay)
        self.actor_optimizer_joint = torch.optim.Adam(self.actor_joint.parameters(),
                                                   lr=self.lr, eps=self.opti_eps,
                                                   weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.critic_optimizer_BS = torch.optim.Adam(self.critic_BS.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.critic_optimizer_MBS = torch.optim.Adam(self.critic_MBS.parameters(),
                                                    lr=self.critic_lr,
                                                    eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        self.critic_optimizer_joint = torch.optim.Adam(self.critic_joint.parameters(),
                                                    lr=self.critic_lr,
                                                    eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        self.kl_dynamic_coeff_network_optimizer = torch.optim.Adam(self.kl_dynamic_coeff_network.parameters(),
                                                                   lr=self.lr, eps=self.opti_eps,
                                                                   weight_decay=self.weight_decay)


    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.actor_optimizer_BS, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        start = time.perf_counter()

        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)
        actions_BS, action_log_probs_BS, rnn_states_actor_BS = self.actor_BS(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)
        actions_MBS, action_log_probs_MBS, rnn_states_actor_MBS = self.actor_MBS(obs,
                                                                             rnn_states_actor,
                                                                             masks,
                                                                             available_actions,
                                                                             deterministic)

        end = time.perf_counter()
        runTime = end - start
        # print(runTime)
        values, q_values, rnn_states_critic = self.critic(cent_obs, actions, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, actions_BS, action_log_probs_BS, rnn_states_actor_BS,\
               actions_MBS, action_log_probs_MBS, rnn_states_actor_MBS, rnn_states_critic

    def get_values(self, cent_obs, actions, rnn_states_critic, masks):
        values, q_values, _ = self.critic(cent_obs, torch.from_numpy(actions).to(device='cuda:0'), rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_actor_BS, rnn_states_actor_MBS, rnn_states_critic, action, action_BS, action_MBS,
                         masks, available_actions=None, active_masks=None):
        # 计算前16个智能体的动作概率和熵
        action_log_probs_1, dist_entropy_1, grad_1 = self.actor.evaluate_actions(obs,
                                                                         rnn_states_actor,
                                                                         action,
                                                                         masks,
                                                                         available_actions if available_actions is not None else None,
                                                                         active_masks if active_masks is not None else None)

        # 计算最后一个智能体（BS）的动作概率和熵
        action_log_probs_2, dist_entropy_2, grad_2 = self.actor_BS.evaluate_actions_BS(obs,
                                                                            rnn_states_actor_BS,
                                                                            action_BS,
                                                                            masks,
                                                                            available_actions if available_actions is not None else None,
                                                                            active_masks if active_masks is not None else None)

        action_log_probs_3, dist_entropy_3, grad_3 = self.actor_MBS.evaluate_actions_MBS(obs,
                                                                                       rnn_states_actor_MBS,
                                                                                       action_MBS,
                                                                                       masks,
                                                                                       available_actions if available_actions is not None else None,
                                                                                       active_masks if active_masks is not None else None)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        elif isinstance(action, np.ndarray):
            action = action
        else:
            raise TypeError("Unsupported type: {}".format(type(action)))
        if isinstance(action_BS, torch.Tensor):
            action_BS = action_BS.detach().cpu().numpy()
        elif isinstance(action_BS, np.ndarray):
            action_BS = action_BS
        else:
            raise TypeError("Unsupported type: {}".format(type(action_BS)))
        if isinstance(action_MBS, torch.Tensor):
            action_MBS = action_MBS.detach().cpu().numpy()
        elif isinstance(action_MBS, np.ndarray):
            action_MBS = action_MBS
        else:
            raise TypeError("Unsupported type: {}".format(type(action_MBS)))

        joint_actions = np.concatenate([action, action_BS, action_MBS], axis=1)
        repeated_action = np.tile(action, (1, 16))  # 在第二个维度（列）重复16次
        joint_actions_2 = np.concatenate([repeated_action, action_BS, action_MBS], axis=1)

        joint_rnn_states = rnn_states_actor + rnn_states_actor_BS + rnn_states_actor_MBS
        joint_action_log_probs, joint_entropy = self.actor_joint.evaluate_actions_joint(obs,
                                                             joint_rnn_states,
                                                             joint_actions,
                                                             masks,
                                                             available_actions if available_actions is not None else None,
                                                             active_masks if active_masks is not None else None)


        # 计算价值函数
        values, q_values, _ = self.critic(cent_obs, action, rnn_states_critic, masks)
        values_BS, q_values_BS, _ = self.critic_BS(cent_obs, action_BS, rnn_states_critic, masks)
        values_MBS, q_values_MBS, _ = self.critic_MBS(cent_obs, action_MBS, rnn_states_critic, masks)
        values_joint, q_values_joint, _ = self.critic_joint(cent_obs, joint_actions_2, rnn_states_critic, masks)

        return values_joint, q_values_joint, values, q_values, values_BS, q_values_BS, values_MBS, q_values_MBS, action_log_probs_1, dist_entropy_1, action_log_probs_2, dist_entropy_2, \
               action_log_probs_3, dist_entropy_3, joint_action_log_probs, joint_entropy, grad_1, grad_2, grad_3

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
