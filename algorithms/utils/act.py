from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.continuous_action = False

        self.continuous_action = True
        action_dim = action_space
        self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        self.action_out_BS = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        self.action_out_joint = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False, noise_scale=0.01):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        action_logit = self.action_out(x)
        actions = action_logit.mode() if deterministic else action_logit.sample()
        # actions = torch.tanh(actions)  # 将动作限制在 [-1, 1] 范围内
        action_log_probs = action_logit.log_probs(actions)

        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # 检查输入数据
        action_log_probs = []
        dist_entropy = []
        # 计算动作分布
        action_logit = self.action_out(x)
        # 检查动作分布参数
        if torch.isnan(action_logit.loc).any() or torch.isinf(action_logit.loc).any():
            raise ValueError("Action distribution loc contains NaN or inf values.")
        if torch.isnan(action_logit.scale).any() or torch.isinf(action_logit.scale).any():
            raise ValueError("Action distribution scale contains NaN or inf values.")
        # 计算动作对数概率
        action_log_probs.append(action_logit.log_probs(action))
        entropy = action_logit.entropy()
        # 计算熵
        if active_masks is not None:
            if len(action_logit.entropy().shape) == len(active_masks.shape):
                dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
            else:
                dist_entropy.append(
                    (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
        else:
            dist_entropy.append(action_logit.entropy().mean())

        # 合并结果
        action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        dist_entropy = dist_entropy[0]  # 取第一个熵值
        grad_ent = torch.autograd.grad(entropy.mean(), [action_logit.loc, action_logit.scale], retain_graph=True, allow_unused=True)

        return action_log_probs, entropy.mean(), grad_ent

    def evaluate_actions_BS(self, x, action, available_actions=None, active_masks=None):

        action_log_probs = []
        dist_entropy = []
        # for action_out, act in zip(self.action_outs, action):
        action_logit = self.action_out_BS(x)
        action_log_probs.append(action_logit.log_probs(action))
        entropy = action_logit.entropy()
        if active_masks is not None:
            if len(action_logit.entropy().shape) == len(active_masks.shape):
                dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
            else:
                dist_entropy.append(
                    (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
        else:
            dist_entropy.append(action_logit.entropy().mean())

        action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        dist_entropy = dist_entropy[0]  # / 2.0 + dist_entropy[1] / 0.98  # ! dosen't make sense
        grad_ent_2 = torch.autograd.grad(entropy.mean(), [action_logit.loc, action_logit.scale], retain_graph=True,
                                       allow_unused=True)

        return action_log_probs, entropy.mean(), grad_ent_2

    def evaluate_actions_joint(self, x, action, available_actions=None, active_masks=None):
        action_log_probs = []
        dist_entropy = []
        # for action_out, act in zip(self.action_outs, action):
        action_logit = self.action_out_joint(x)
        action_log_probs.append(action_logit.log_probs(action))
        entropy = action_logit.entropy()
        if active_masks is not None:
            if len(action_logit.entropy().shape) == len(active_masks.shape):
                dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
            else:
                dist_entropy.append(
                    (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
        else:
            dist_entropy.append(action_logit.entropy().mean())

        action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        dist_entropy = dist_entropy[0]  # / 2.0 + dist_entropy[1] / 0.98  # ! dosen't make sense

        return action_log_probs, entropy
