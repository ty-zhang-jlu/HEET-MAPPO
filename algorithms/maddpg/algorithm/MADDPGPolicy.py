import torch
import numpy as np
from torch.distributions import OneHotCategorical
from global_matd3.algorithms.base.mlp_policy import MLPPolicy
from global_matd3.algorithms.maddpg.algorithm.actor_critic import MADDPG_Actor, MADDPG_Critic
from global_matd3.algorithms.matd3.algorithm.actor_critic import MATD3_Actor, MATD3_Critic
from global_matd3.utils.util import get_dim_from_space, DecayThenFlatSchedule, soft_update, \
    hard_update, \
    gumbel_softmax, onehot_from_logits, gaussian_noise, avail_choose, to_numpy


class MADDPGPolicy(MLPPolicy):
    """
    MADDPG/MATD3 Policy Class to wrap actor/critic and compute actions. See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param target_noise: (int) std of target smoothing noise to add for MATD3 (applies only for continuous actions)
    :param td3: (bool) whether to use MATD3 or MADDPG.
    :param train: (bool) whether the policy will be trained.
    """

    def __init__(self, config, policy_config, act_dim, target_noise=None, td3=True, train=True):
        self.config = config
        self.device = config['device']
        self.args = self.config["args"]
        self.tau = self.args.tau
        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        # self.central_obs_dim, self.central_act_dim = policy_config["cent_obs_dim"], policy_config["cent_act_dim"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = act_dim
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.target_noise = target_noise
        self.N = 16
        self.M = 4
        self.K = 5
        self.num_agents = self.N + 1
        self.central_act_dim = 2 * (self.num_agents - 1) + 2 * self.M * self.K


        actor_class = MATD3_Actor if td3 else MADDPG_Actor
        critic_class = MATD3_Critic if td3 else MADDPG_Critic

        self.actor = actor_class(self.args, self.obs_dim, self.act_dim, self.device)
        self.actor_cen = actor_class(self.args, self.obs_dim, self.central_act_dim, self.device)
        self.target_actor = actor_class(self.args, self.obs_dim, self.act_dim, self.device)
        self.target_actor_cen = actor_class(self.args, self.obs_dim, self.central_act_dim, self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor_cen.load_state_dict(self.actor_cen.state_dict())

        # self.critic = critic_class(self.args, self.central_obs_dim, self.central_act_dim, self.device)
        # self.target_critic = critic_class(self.args, self.central_obs_dim, self.central_act_dim, self.device)
        self.critic = critic_class(self.args, self.obs_dim, self.central_act_dim, self.device)
        self.target_critic = critic_class(self.args, self.obs_dim, self.central_act_dim, self.device)

        # sync the target weights
        self.target_critic.load_state_dict(self.critic.state_dict())

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.actor_cen_optimizer = torch.optim.Adam(self.actor_cen.parameters(), lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)
            self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish,
                                                         self.args.epsilon_anneal_time, decay="linear")

    def gaussian_noise(self, tensor_shape, std, device):
        noise = torch.randn(tensor_shape, device=device) * std
        return noise

    def get_actions(self, obs, available_actions=None, t_env=None, explore=True, use_target=False, use_gumbel=False):
        """See parent class."""
        # print(obs)
        # batch_size = obs.shape[0]
        eps = None
        if use_target:
                actor_out = self.target_actor(obs)
        else:
                actor_out = self.actor(obs)
        if explore:
            action = tuple(gaussian_noise(out.shape, self.args.act_noise_std, device=self.device) + out for out in actor_out)
            action = action
        elif use_target and self.target_noise is not None:
            assert isinstance(self.target_noise, float)
            action = gaussian_noise(actor_out[0][0].shape, self.target_noise, device=self.device) + actor_out[0][0]
        else:
            action = actor_out

        return action, eps

    def get_cen_actions(self, obs, available_actions=None, t_env=None, explore=True, use_target=False, use_gumbel=False):
        """See parent class."""
        # print(obs)
        # batch_size = obs.shape[0]
        eps = None
        if use_target:
                actor_out = self.target_actor_cen(obs)
        else:
                actor_out = self.actor_cen(obs)
        if explore:
            action = tuple(gaussian_noise(out.shape, self.args.act_noise_std, device=self.device) + out for out in actor_out)
            action = action[0]
        elif use_target and self.target_noise is not None:
            assert isinstance(self.target_noise, float)
            action = gaussian_noise(actor_out[0][0].shape, self.target_noise, device=self.device) + actor_out[0][0]
        else:
            action = actor_out

        return action, eps

    def get_random_actions(self, agent_act_dim, available_actions=None):
        """See parent class."""
        random_actions = np.random.uniform(-1, 1,
                                           size=(1, agent_act_dim))

        return random_actions

    def soft_target_updates(self):
        """Polyal update the target networks."""
        # polyak updates to target networks
        soft_update(self.target_critic, self.critic, self.args.tau)
        soft_update(self.target_actor, self.actor, self.args.tau)

    def hard_target_updates(self):
        """Copy the live networks into the target networks."""
        # polyak updates to target networks
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)
