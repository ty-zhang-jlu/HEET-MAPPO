"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_actor_critic.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from global_mappo_8.algorithms.utils.util import init, check
from global_mappo_8.algorithms.utils.cnn import CNNBase
from global_mappo_8.algorithms.utils.mlp import MLPBase
from global_mappo_8.algorithms.utils.rnn import RNNLayer
from global_mappo_8.algorithms.utils.act import ACTLayer
from global_mappo_8.algorithms.utils.popart import PopArt
from global_mappo_8.utils.util import get_shape_from_obs_space
import numpy as np


class CooperativeAwareModule(nn.Module):
    def __init__(self, input_dim, embed_dim, num_agents, recurrent_N, hidden_size, device):
        super(CooperativeAwareModule, self).__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.recurrent_N = recurrent_N
        self.hidden_size = hidden_size
        self.device = device

        # RNN编码器用于历史策略熵序列
        self.rnn = nn.GRU(input_dim, hidden_size, recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # MLP编码器（共享参数）
        self.mlp_encoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        # 注意力机制的投影矩阵
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, policy_entropies_history, agent_id):
        # 确保输入形状正确
        if policy_entropies_history.dim() == 2:
            # [seq_len, num_agents] -> [seq_len, num_agents, 1]
            policy_entropies_history = policy_entropies_history.unsqueeze(-1)
        seq_len, num_agents, input_dim = policy_entropies_history.shape
        # print(f"seq_len: {seq_len}, num_agents: {num_agents}, input_dim: {input_dim}")

        # 检查agent_id是否在有效范围内
        if agent_id < 0 or agent_id >= num_agents:
            # print(f"Warning: agent_id {agent_id} out of range [0, {num_agents - 1}]. Using agent_id 0.")
            agent_id = 0

        # 初始化RNN隐藏状态
        hxs = torch.zeros(self.recurrent_N, num_agents, self.hidden_size).to(self.device)
        # print(f"Hidden state shape: {hxs.shape}")

        # 通过RNN编码
        rnn_out, _ = self.rnn(policy_entropies_history, hxs)
        # print(f"RNN output shape: {rnn_out.shape}")

        # 取最后一个时间步的输出
        rnn_out_last = rnn_out[-1]  # [num_agents, hidden_size]
        # print(f"Last RNN output shape: {rnn_out_last.shape}")

        # 2. MLP编码
        policy_embeddings = self.mlp_encoder(rnn_out_last)  # [num_agents, embed_dim]
        # print(f"Policy embeddings shape: {policy_embeddings.shape}")
        # print(f"DEBUG: policy_embeddings shape: {policy_embeddings.shape}")
        # print(f"DEBUG: policy_embeddings: {policy_embeddings}")
        # 修复：检查policy_embeddings是否为空
        if policy_embeddings.numel() == 0:
            # 返回零向量作为备用
            print("MLP编码后的向量为空")
            context_vector = torch.zeros(self.embed_dim).to(self.device)
            attention_weights = torch.zeros(num_agents).to(self.device)
            return context_vector, attention_weights
        # 3. 注意力机制
        # 确保agent_id在有效范围内
        if agent_id >= policy_embeddings.size(0):
            # print(f"Agent_id {agent_id} out of policy_embeddings range. Using 0.")
            agent_id = 0

        # 获取查询向量
        query_agent_embedding = policy_embeddings[0][agent_id]  # [embed_dim]
        # print(f"Query agent embedding shape: {query_agent_embedding.shape}")
        # print(f"Query agent embedding sum: {query_agent_embedding.sum()}")

        if query_agent_embedding.numel() == 0:
            # print("ERROR: Query agent embedding is empty after indexing!")
            # 使用第一个智能体
            if policy_embeddings.size(0) > 0:
                query_agent_embedding = policy_embeddings[0]
                # print(f"Using first agent embedding instead: {query_agent_embedding.shape}")

        # 通过W_q变换
        query_transformed = self.W_q(query_agent_embedding)
        # print(f"Query transformed shape: {query_transformed.shape}")
        # print(f"Query transformed sum: {query_transformed.sum()}")

        query = query_transformed.unsqueeze(0)  # [1, embed_dim]
        # print(f"Final query shape: {query.shape}")

        keys = self.W_k(policy_embeddings)  # [num_agents, embed_dim]
        # print(f"Keys shape: {keys.shape}")

        # print(f"=== DEBUG END ===")
        # print(f"Query shape: {query.shape}, Keys shape: {keys.shape}")

        # 检查查询向量是否为空
        if query.numel() == 0:
            print("Warning: Query vector is empty. Using zero vector.")
            query = torch.zeros(1, self.embed_dim).to(self.device)

        # 计算注意力得分
        attention_scores = torch.matmul(query, keys.transpose(0, 1))  # [1, num_agents]
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, num_agents]
        # print(f"Attention weights shape: {attention_weights.shape}")

        # 使用value投影
        values = self.W_v(policy_embeddings)  # [num_agents, embed_dim]
        # print(f"Values shape: {values.shape}")

        # 加权求和得到协作上下文向量
        context_vector = torch.matmul(attention_weights, values)  # [1, embed_dim]
        # print(f"Context vector before squeeze: {context_vector.shape}")

        # 确保返回正确维度的张量
        context_vector = context_vector.squeeze(0)  # [embed_dim]
        # print(f"Final context vector shape: {context_vector.shape}")

        return context_vector, attention_weights.squeeze(0)


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents
        self.device = device

        # 添加历史策略熵队列
        self.history_length = args.history_length  # 需要添加这个参数，例如10
        # 修改初始化部分
        self.history_entropies = torch.zeros(self.history_length, self.num_agents,
                                             device=device).clone().detach()
        self.history_entropies.requires_grad_(False)  # 确保不需要梯度

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape[0])
        # base = MLPBase
        self.base = base(args, obs_shape[0])

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # 修改CAM模块，使其包括RNN
        self.cam = CooperativeAwareModule(input_dim=1, embed_dim=32, num_agents=self.num_agents,
                                            recurrent_N=args.recurrent_N, hidden_size=args.hidden_size, device=device)
        # 添加适配层，将融合后的特征映射回原来的 hidden_size
        self.feature_adapter = nn.Linear(self.hidden_size, self.hidden_size)
        # 修改策略网络的输入维度
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)  # +64是CAM的输出维度

        self.to(device)


    def forward(self, obs, rnn_states, masks, available_actions=None, agent_id=0, deterministic=False):
        try:
            obs = check(obs).to(**self.tpdv)
            rnn_states = check(rnn_states).to(**self.tpdv)
            masks = check(masks).to(**self.tpdv)
            if available_actions is not None:
                available_actions = check(available_actions).to(**self.tpdv)

            actor_features = self.base(obs)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

            context_vector, attention_weights = self.cam(self.history_entropies, agent_id)
            # print("协作上下文向量", context_vector)
            # print("注意力权重：", attention_weights)

            # 特征融合：将上下文向量与actor特征拼接
            batch_size = actor_features.size(0)
            # 扩展context_vector以匹配批次大小
            if context_vector.dim() == 1:  # [embed_dim]
                context_vector = context_vector.unsqueeze(0).expand(batch_size, -1)  # [batch_size, embed_dim]

            # 检查维度是否匹配
            if actor_features.dim() != context_vector.dim():
                raise ValueError(
                    f"Dimension mismatch: actor_features {actor_features.dim()}D, context_vector {context_vector.dim()}D")
            # print("动作特征", actor_features)
            # fused_features = torch.cat([actor_features, context_vector], dim=-1)
            # print("融合特征", fused_features)
            # 确保维度匹配后才能相加
            # print(actor_features.shape)
            # print(context_vector.shape)
            fused_features = actor_features + 0.5 * context_vector

            # 通过适配层映射回原来的维度
            # fused_features = self.feature_adapter(fused_features)

            # actions, action_log_probs, dist_entropy = self.act(fused_features, available_actions, deterministic)
            actions, action_log_probs, dist_entropy = self.act(fused_features, available_actions, deterministic)
            # 更新历史熵队列
            if dist_entropy.dim() == 0:  # 标量
                dist_entropy = dist_entropy.unsqueeze(0)

            # 更新历史熵队列
            new_entropy = dist_entropy.detach()

            # 调试信息
            # print(f"=== DEBUG ===")
            # print(f"Current history: {self.history_entropies}")

            # 确保取的是新值
            entropy_value = new_entropy[0, 0].clone()  # 使用clone确保是新值
            # print(f"New entropy value to add: {entropy_value:.4f}")

            with torch.no_grad():
                new_history = torch.zeros_like(self.history_entropies)

                # 向前滚动：将第1行到最后一行复制到新历史的前n-1行
                if self.history_entropies.size(0) > 1:
                    new_history[:-1] = self.history_entropies[1:].clone()

                # 在最后一行设置新值
                new_history[-1] = entropy_value

                # 替换原历史张量
                self.history_entropies = new_history

            # print(f"Updated history: {self.history_entropies}")
            # print(f"=== DEBUG END ===")

            return actions, action_log_probs, rnn_states, dist_entropy, attention_weights

        except Exception as e:
            print(f"Error in R_Actor.forward: {e}")
            print("Returning zero tensors as fallback")

            # 返回零张量作为备用
            batch_size = obs.size(0) if obs.dim() > 1 else 1
            # 正确获取动作维度
            action_dim = self.act.get_action_dim()

            actions = torch.zeros(batch_size, action_dim).to(self.device)
            action_log_probs = torch.zeros(batch_size, 1).to(self.device)
            rnn_states = torch.zeros_like(rnn_states) if rnn_states is not None else torch.zeros(1,
                                                                                                 self.hidden_size).to(
                self.device)
            dist_entropy = torch.zeros(1).to(self.device)

            return actions, action_log_probs, rnn_states, dist_entropy, attention_weights


    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, grad_1 = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy, grad_1

    def evaluate_actions_BS(self, obs, rnn_states_BS, action_BS, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states_BS).to(**self.tpdv)
        action_BS = check(action_BS).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, grad_2 = self.act.evaluate_actions_BS(actor_features,
                                                                   action_BS, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy, grad_2

    def evaluate_actions_MBS(self, obs, rnn_states_MBS, action_MBS, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states_MBS).to(**self.tpdv)
        action_MBS = check(action_MBS).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, grad_3 = self.act.evaluate_actions_MBS(actor_features,
                                                                   action_MBS, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy, grad_3

    def evaluate_actions_joint(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions_joint(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, action_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # 参数噪声配置
        self.use_param_noise = args._use_param_noise
        self.param_noise_std = args._param_noise_std
        self.perturbed_weights = {}  # 存储扰动后的参数

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape[0])

        total_shape = cent_obs_shape[0] + action_space
        # print(f"Critic total input shape: {total_shape}")
        self.base_agent = base(args, total_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
            self.q_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
            self.q_out = nn.Linear(self.hidden_size, 1)

        self.to(device)

    def forward(self, cent_obs, actions, rnn_states, masks):
        # print(f"Critic input - cent_obs shape: {cent_obs.shape}")
        # print(f"Critic input - actions shape: {actions.shape}")
        # 使用扰动参数（如果启用）
        if self.use_param_noise:
            for name, param in self.named_parameters():
                if name in self.perturbed_weights:
                    param.data = self.perturbed_weights[name]

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device='cuda:0')  # 转换为 CUDA 张量
        if isinstance(cent_obs, np.ndarray):
            cent_obs = torch.from_numpy(cent_obs).to(device='cuda:0')  # 转换为 CUDA 张量

        # 确保动作和观测的批次大小一致
        if cent_obs.shape[0] != actions.shape[0]:
            # 如果批次大小不匹配，调整动作的批次大小
            if actions.shape[0] == 1 and cent_obs.shape[0] > 1:
                actions = actions.expand(cent_obs.shape[0], -1)
            else:
                raise ValueError(f"Batch size mismatch: cent_obs {cent_obs.shape[0]}, actions {actions.shape[0]}")

        # 拼接观测和动作
        inputs = torch.cat([cent_obs, actions], dim=-1)
        # print(f"Critic concatenated inputs shape: {inputs.shape}")

        critic_features = self.base(cent_obs)

        inputs = torch.cat([cent_obs, actions], dim=-1)
        action_features = self.base_agent(inputs)
        # print(f"Critic features shape: {critic_features.shape}")
        # print(f"Action features shape: {action_features.shape}")
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            action_features, rnn_states_q = self.rnn(action_features, rnn_states, masks)
        values = self.v_out(critic_features)
        # 新增Q值计算
        q_values = self.q_out(action_features)
        # values = torch.mean(q_values, dim=0)

        return values, q_values, rnn_states

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
