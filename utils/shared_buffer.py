import torch
import numpy as np
from global_mappo.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        obs_shape = get_shape_from_obs_space(obs_space)
        # share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        share_obs_shape = get_shape_from_obs_space(obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_BS = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_MBS = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape[0]), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape[0]), dtype=np.float32)
        self.actions_BS = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape[1]), dtype=np.float32)
        self.action_log_probs_BS = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape[1]), dtype=np.float32)
        self.actions_MBS = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape[2]), dtype=np.float32)
        self.action_log_probs_MBS = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape[2]), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        # 新增exploitation网络字段（与exploration维度完全对齐）
        self.actions_exploit = np.zeros_like(self.actions)  # [episode_length, n_rollout_threads, num_agents, act_shape]
        self.actions_BS_exploit = np.zeros_like(self.actions_BS)
        self.action_log_probs_exploit = np.zeros_like(self.action_log_probs)
        self.action_log_probs_BS_exploit = np.zeros_like(self.action_log_probs)
        self.value_preds_exploit = np.zeros_like(
            self.value_preds)  # [episode_length + 1, n_rollout_threads, num_agents, 1]
        # Exploitation网络RNN状态（即使不用RNN也需占位）
        self.rnn_states_exploit = np.zeros_like(self.rnn_states)  # [T+1, n_threads, n_agents, ...]
        self.rnn_states_BS_exploit = np.zeros_like(self.rnn_states_BS)
        self.rnn_states_critic_exploit = np.zeros_like(self.rnn_states_critic)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_actor_BS, rnn_states_actor_MBS, rnn_states_critic, actions, actions_BS, actions_MBS,
               action_log_probs, action_log_probs_BS, action_log_probs_MBS, value_preds,  rewards, masks,
                actions_exploit=None, actions_BS_exploit=None, action_log_probs_exploit=None, action_log_probs_BS_exploit=None,
               value_preds_exploit=None, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_BS[self.step + 1] = rnn_states_actor_BS.copy()
        self.rnn_states_MBS[self.step + 1] = rnn_states_actor_MBS.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.actions_BS[self.step] = actions_BS.copy()
        self.actions_MBS[self.step] = actions_MBS.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.action_log_probs_BS[self.step] = action_log_probs_BS.copy()
        self.action_log_probs_MBS[self.step] = action_log_probs_MBS.copy()
        self.value_preds[self.step] = value_preds.copy()
        # 存储exploitation数据（新增逻辑）
        if actions_exploit is not None:
            self.actions_exploit[self.step] = actions_exploit.copy()
        if action_log_probs_exploit is not None:
            self.action_log_probs_exploit[self.step] = action_log_probs_exploit.copy()
        if actions_BS_exploit is not None:
            self.actions_BS_exploit[self.step] = actions_BS_exploit.copy()
        if action_log_probs_BS_exploit is not None:
            self.action_log_probs_BS_exploit[self.step] = action_log_probs_BS_exploit.copy()
        if value_preds_exploit is not None:
            self.value_preds_exploit[self.step] = value_preds_exploit.copy()
        # 存储exploitation的RNN状态（零值占位）
        self.rnn_states_exploit[
            self.step + 1] = rnn_states_actor.copy() if rnn_states_actor is not None else np.zeros_like(
            self.rnn_states_exploit[0])
        self.rnn_states_BS_exploit[
            self.step + 1] = rnn_states_actor_BS.copy() if rnn_states_actor_BS is not None else np.zeros_like(
            self.rnn_states_BS_exploit[0])
        self.rnn_states_critic_exploit[
            self.step + 1] = rnn_states_critic.copy() if rnn_states_critic is not None else np.zeros_like(
            self.rnn_states_critic_exploit[0])
        self.rewards[self.step] = rewards.copy()
        padding = np.ones((2, 1, 1))
        masks_padded = np.concatenate([masks, padding], axis=1)
        self.masks[self.step + 1] = masks_padded.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_BS, rnn_states_MBS, rnn_states_critic, actions, actions_BS, actions_MBS,
                     action_log_probs, action_log_probs_BS, action_log_probs_MBS, value_preds, rewards, masks,
                     actions_exploit=None, actions_BS_exploit=None, action_log_probs_exploit=None,
                     action_log_probs_BS_exploit=None, value_preds_exploit=None, bad_masks=None,
                     active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_BS[self.step + 1] = rnn_states_BS.copy()
        self.rnn_states_MBS[self.step + 1] = rnn_states_MBS.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.actions_BS[self.step] = actions_BS.copy()
        self.actions_MBS[self.step] = actions_MBS.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.action_log_probs_BS[self.step] = action_log_probs_BS.copy()
        self.action_log_probs_MBS[self.step] = action_log_probs_MBS.copy()
        # 新增exploitation数据存储
        if actions_exploit is not None:
            self.actions_exploit[self.step] = actions_exploit.copy()
        if action_log_probs_exploit is not None:
            self.action_log_probs_exploit[self.step] = action_log_probs_exploit.copy()
        if actions_BS_exploit is not None:
            self.actions_BS_exploit[self.step] = actions_BS_exploit.copy()
        if action_log_probs_BS_exploit is not None:
            self.action_log_probs_BS_exploit[self.step] = action_log_probs_BS_exploit.copy()
        if value_preds_exploit is not None:
            self.value_preds_exploit[self.step] = value_preds_exploit.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_BS[0] = self.rnn_states_BS[-1].copy()
        self.rnn_states_MBS[0] = self.rnn_states_MBS[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_BS[0] = self.rnn_states_BS[-1].copy()
        self.rnn_states_MBS[0] = self.rnn_states_MBS[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # 仅返回exploitation数据，RNN状态设为空或零
        rnn_states_exploit = np.zeros_like(self.rnn_states_exploit[:-1])  # 填充零值
        rnn_states_BS_exploit = np.zeros_like(self.rnn_states_BS_exploit[:-1])
        rnn_states_critic_exploit = np.zeros_like(self.rnn_states_critic_exploit[:-1])

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        actions_exploit = self.actions_exploit.reshape(-1, self.actions_exploit.shape[-1])
        actions_BS_exploit = self.actions_BS_exploit.reshape(-1, self.actions_BS_exploit.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs_exploit = self.action_log_probs_exploit.reshape(-1, self.action_log_probs_exploit.shape[-1])
        action_log_probs_BS_exploit = self.action_log_probs_BS_exploit.reshape(-1, self.action_log_probs_BS_exploit.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch_exploit = actions_exploit[indices]
            actions_BS_batch_exploit = actions_BS_exploit[indices]
            rnn_states_batch_exploit = rnn_states_exploit[indices]
            rnn_states_BS_batch_exploit = rnn_states_BS_exploit[indices]
            rnn_states_critic_batch_exploit = rnn_states_critic_exploit[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch_exploit = action_log_probs_exploit[indices]
            old_action_log_probs_BS_batch_exploit = action_log_probs_BS_exploit[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch_exploit, rnn_states_BS_batch_exploit, rnn_states_critic_batch_exploit, actions_batch_exploit, actions_BS_batch_exploit, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch_exploit, old_action_log_probs_BS_batch_exploit, \
                  adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_BS = self.rnn_states_BS[:-1].reshape(-1, *self.rnn_states_BS.shape[3:])
        rnn_states_MBS = self.rnn_states_MBS[:-1].reshape(-1, *self.rnn_states_MBS.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        actions_BS = self.actions_BS.reshape(-1, self.actions_BS.shape[-1])
        actions_MBS = self.actions_MBS.reshape(-1, self.actions_MBS.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        action_log_probs_BS = self.action_log_probs_BS.reshape(-1, self.action_log_probs_BS.shape[-1])
        action_log_probs_MBS = self.action_log_probs_MBS.reshape(-1, self.action_log_probs_MBS.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_BS_batch = []
            rnn_states_MBS_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            actions_BS_batch = []
            actions_MBS_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            old_action_log_probs_BS_batch = []
            old_action_log_probs_MBS_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_BS_batch.append(rnn_states_BS[0:1, ind])
                rnn_states_MBS_batch.append(rnn_states_MBS[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                actions_BS_batch.append(actions_BS[:, ind])
                actions_MBS_batch.append(actions_MBS[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                old_action_log_probs_BS_batch.append(action_log_probs_BS[:, ind])
                old_action_log_probs_MBS_batch.append(action_log_probs_MBS[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            actions_BS_batch = np.stack(actions_BS_batch, 1)
            actions_MBS_batch = np.stack(actions_MBS_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            old_action_log_probs_BS_batch = np.stack(old_action_log_probs_BS_batch, 1)
            old_action_log_probs_MBS_batch = np.stack(old_action_log_probs_MBS_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_BS_batch = np.stack(rnn_states_BS_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_MBS_batch = np.stack(rnn_states_MBS_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            actions_BS_batch = _flatten(T, N, actions_BS_batch)
            actions_MBS_batch = _flatten(T, N, actions_MBS_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            old_action_log_probs_BS_batch = _flatten(T, N, old_action_log_probs_BS_batch)
            old_action_log_probs_MBS_batch = _flatten(T, N, old_action_log_probs_MBS_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_BS_batch, rnn_states_MBS_batch, rnn_states_critic_batch, actions_batch, actions_BS_batch, \
                  actions_MBS_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, old_action_log_probs_BS_batch, \
                  old_action_log_probs_MBS_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        actions_BS = _cast(self.actions_BS)
        actions_MBS = _cast(self.actions_MBS)
        action_log_probs = _cast(self.action_log_probs)
        action_log_probs_BS = _cast(self.action_log_probs_BS)
        action_log_probs_MBS = _cast(self.action_log_probs_MBS)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_BS = self.rnn_states_BS[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_BS.shape[3:])
        rnn_states_MBS = self.rnn_states_MBS[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_MBS.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_BS_batch = []
            rnn_states_MBS_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            actions_BS_batch = []
            actions_MBS_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            old_action_log_probs_BS_batch = []
            old_action_log_probs_MBS_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                actions_BS_batch.append(actions_BS[ind:ind + data_chunk_length])
                actions_MBS_batch.append(actions_MBS[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                old_action_log_probs_BS_batch.append(action_log_probs_BS[ind:ind + data_chunk_length])
                old_action_log_probs_MBS_batch.append(action_log_probs_MBS[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_BS_batch.append(rnn_states_BS[ind])
                rnn_states_MBS_batch.append(rnn_states_MBS[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            actions_BS_batch = np.stack(actions_BS_batch, axis=1)
            actions_MBS_batch = np.stack(actions_MBS_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            old_action_log_probs_BS_batch = np.stack(old_action_log_probs_BS_batch, axis=1)
            old_action_log_probs_MBS_batch = np.stack(old_action_log_probs_MBS_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_BS_batch = np.stack(rnn_states_BS_batch).reshape(N, *self.rnn_states_BS.shape[3:])
            rnn_states_MBS_batch = np.stack(rnn_states_MBS_batch).reshape(N, *self.rnn_states_MBS.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            actions_BS_batch = _flatten(L, N, actions_BS_batch)
            actions_MBS_batch = _flatten(L, N, actions_MBS_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            old_action_log_probs_BS_batch = _flatten(L, N, old_action_log_probs_BS_batch)
            old_action_log_probs_MBS_batch = _flatten(L, N, old_action_log_probs_MBS_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_BS_batch, rnn_states_MBS_batch, rnn_states_critic_batch, actions_batch, actions_BS_batch, \
                  actions_MBS_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, old_action_log_probs_BS_batch, \
                  old_action_log_probs_MBS_batch, adv_targ, available_actions_batch
