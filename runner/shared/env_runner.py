import time
import numpy as np
import torch
from global_mappo_1.runner.shared.base_runner import Runner
from global_mappo_1.algorithms.algorithm.OPT_DDPG import Agent_J
from scipy.io import savemat
import csv
import os


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.M = 4
        self.J = 5
        self.N = 16
        self.K = 5
        self.n_input_j = 1
        self.n_output_j = 2 * self.J * self.K
        self.alpha = 0.0001
        self.beta = 0.001
        self.tau = 0.005
        self.n_agents = self.N + 1
        self.whether_exploit = False

    def run(self):
        agent_j = Agent_J(self.alpha, self.beta, self.n_input_j, self.tau, self.n_output_j, gamma=0.99, max_size=10000,
                          C_fc1_dims=512,
                          C_fc2_dims=256, C_fc3_dims=128, A_fc1_dims=256, A_fc2_dims=128, batch_size=64, n_agents=1)
        obs, obs_j = self.envs.envs[0].env.reset()
        self.warmup(obs)

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        reward_list = []
        rate_list = []
        jain_list = []
        kl_1_list = []
        lr_1_list = []
        v_L_list = []
        Bup_1_list = []
        Q_error_list = []
        Noise_list = []
        Cei_1_list = []
        Cei_2_list = []
        J_Entropy_list = []
        Ers_1_list = []
        Ers_2_list = []
        user_rate_1_list = []
        user_rate_2_list = []
        user_rate_3_list = []
        user_rate_4_list = []
        user_rate_5_list = []
        powerin_list = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            done = False
            self.epsilon = 0
            vary_L = 10.0
            v_L_list.append(vary_L)
            vary_noise = 0
            Noise_list.append(vary_noise)

            for step in range(self.episode_length):
                rewards = []
                rates = []
                jain_fairnesss = []
                rate_1 = []
                rate_2 = []
                rate_3 = []
                rate_4 = []
                rate_5 = []
                powers = []
                (
                    value_explore,
                    action_expolit,
                    action_log_probs_expolit,
                    rnn_states,
                    rnn_states_critic,
                    actions_BS_expolit,
                    action_log_probs_BS_expolit,
                    action_explore,
                    action_log_probs_explore,
                    actions_BS_explore,
                    action_log_probs_BS_explore,
                    rnn_states_BS,
                    action_G,
                    action_phase,
                    action_tau,
                    value_expolit,
                    self.whether_exploit
                ) = self.collect(step)

                action_j = agent_j.choose_action(np.asarray(obs_j).flatten())
                w_theta = action_j[0: self.J * self.K] * np.pi
                w_beta = (action_j[self.J * self.K: 2 * self.J * self.K] + 1) / 2
                w_array = np.cos(w_theta) * w_beta + np.sin(w_theta) * w_beta * 1j
                action_W = np.reshape(w_array, (self.J, self.K))
                obs_new, reward, dones, infos, obs_j_new, reward_j, sum_rate, jain_fairness, rate_array, power_in = self.envs.envs[0].env.step(action_G, action_phase, np.array(action_tau), action_W)
                rewards.append(reward)
                rates.append(sum_rate)
                jain_fairnesss.append(jain_fairness)
                rate_1.append(rate_array[0])
                rate_2.append(rate_array[1])
                rate_3.append(rate_array[2])
                rate_4.append(rate_array[3])
                rate_5.append(rate_array[4])
                powers.append(power_in)

                if step == self.episode_length - 1:
                    dones = [True for _ in range(self.n_agents)]
                    done = True

                data = (obs_new, reward, dones, infos, action_explore, action_log_probs_explore, rnn_states, actions_BS_explore, action_log_probs_BS_explore,
                    rnn_states_BS, value_explore, rnn_states_critic, action_expolit, actions_BS_expolit, action_log_probs_expolit,
                           action_log_probs_BS_expolit, value_expolit)
                self.insert(data)
                agent_j.remember(np.asarray(obs_j).flatten(), np.asarray(action_j).flatten(),
                                 reward_j, np.asarray(obs_j_new).flatten(), done)
            if episode >= 0:
                self.compute()
                if episode <= 4000:
                    train_infos = self.train_1(vary_L, vary_noise)
                else:
                    train_infos = self.train_2(vary_L, vary_noise)
                agent_j.learn()

                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                train_infos["average_episode_rewards"] = np.mean(rewards)
                self.log_train(train_infos, total_num_steps)
                kl_1_list.append(train_infos['kl'])
                lr_1_list.append(train_infos['lr_1'])
                Bup_1_list.append(train_infos['Bup_1'])
                Q_error_list.append(train_infos['Q_err'])
                Cei_1_list.append(train_infos['Cei_1'])
                Cei_2_list.append(train_infos['Cei_2'])
                J_Entropy_list.append(train_infos['J_T'])
                Ers_1_list.append(train_infos['Ers_1'])
                Ers_2_list.append(train_infos['Ers_2'])

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()
                self.save_exploit()

            if episode % self.log_interval == 0:
                end = time.time()

                reward_list.append(np.mean(rewards))
                rate_list.append(np.mean(rates))
                jain_list.append(np.mean(jain_fairnesss))
                user_rate_1_list.append(np.mean(rate_1))
                user_rate_2_list.append(np.mean(rate_2))
                user_rate_3_list.append(np.mean(rate_3))
                user_rate_4_list.append(np.mean(rate_4))
                user_rate_5_list.append(np.mean(rate_5))
                powerin_list.append(np.mean(powers))

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        file_name = 'HMAPPO_reward_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in reward_list:
                writer.writerow([item])
        file_name = 'HMAPPO_jain_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in jain_list:
                writer.writerow([item])
        file_name = 'HMAPPO_rate1_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in user_rate_1_list:
                writer.writerow([item])
        file_name = 'HMAPPO_rate2_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in user_rate_2_list:
                writer.writerow([item])
        file_name = 'HMAPPO_rate3_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in user_rate_3_list:
                writer.writerow([item])
        file_name = 'HMAPPO_rate4_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in user_rate_4_list:
                writer.writerow([item])
        file_name = 'HMAPPO_rate5_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in user_rate_5_list:
                writer.writerow([item])
        file_name = 'HMAPPO_power_35.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in powerin_list:
                writer.writerow([item])

    def warmup(self, obs):
        if self.use_centralized_V:
            obs = np.array(obs)
            share_obs = np.repeat(obs[np.newaxis, :, :], self.n_rollout_threads, axis=0)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs[:, :, :10].copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        self.trainer_exploit.prep_rollout()

        (
                value_expolit,
                action_expolit,
                action_log_prob_expolit,
                rnn_states,
                action_BS_expolit,
                action_log_probs_BS_expolit,
                rnn_states_BS,
                rnn_states_critic
        ) = self.trainer_exploit.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
        )

        (
                value_explore,
                action_explore,
                action_log_prob_explore,
                rnn_states,
                action_BS_explore,
                action_log_probs_BS_explore,
                rnn_states_BS,
                rnn_states_critic,
        ) = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
        )

        value = value_explore + self.epsilon * value_expolit
        action = action_explore + self.epsilon * action_expolit
        action_BS = action_BS_explore + self.epsilon * action_BS_expolit

        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        rnn_states_BS = np.array(
            np.split(_t2n(rnn_states_BS), self.n_rollout_threads))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        value_expolit = np.array(np.split(_t2n(value_expolit), self.n_rollout_threads))
        value_explore = np.array(np.split(_t2n(value_explore), self.n_rollout_threads))

        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        actions_BS = np.array(np.split(_t2n(action_BS), self.n_rollout_threads))

        action_expolit = np.array(np.split(_t2n(action_expolit), self.n_rollout_threads))
        action_log_probs_expolit = np.array(
            np.split(_t2n(action_log_prob_expolit), self.n_rollout_threads)
        )
        actions_BS_expolit = np.array(np.split(_t2n(action_BS_expolit), self.n_rollout_threads))
        action_log_probs_BS_expolit = np.array(
            np.split(_t2n(action_log_probs_BS_expolit), self.n_rollout_threads)
        )

        action_explore = np.array(np.split(_t2n(action_explore), self.n_rollout_threads))
        action_log_probs_explore = np.array(
            np.split(_t2n(action_log_prob_explore), self.n_rollout_threads)
        )
        actions_BS_explore = np.array(np.split(_t2n(action_BS_explore), self.n_rollout_threads))
        action_log_probs_BS_explore = np.array(
            np.split(_t2n(action_log_probs_BS_explore), self.n_rollout_threads)
        )

        actions_env = actions
        actions_env_BS = actions_BS
        action_phase = []
        action_tau = []
        actor_action = []
        for j in range(self.n_agents):
            if j == self.n_agents - 1:
                action_BS = np.clip(actions_env_BS[0, j, :], -0.999, 0.999).reshape(-1)
                g_theta = action_BS[0: self.M * self.K] * np.pi
                actor_action.extend(g_theta.tolist())
                g_beta = (action_BS[self.M * self.K: 2 * self.M * self.K] + 1) / 2
                actor_action.extend(g_beta.tolist())
                g_array = np.cos(g_theta) * g_beta + np.sin(g_theta) * g_beta * 1j
                action_G = np.reshape(g_array, (self.M, self.K))

            else:
                action_phase.append(np.floor(((actions_env[0, j, 0] + 1) / 2) * (2 ** 3 + 1)) * (2 * np.pi / (2 ** 3)))
                actor_action.append(np.floor(((actions_env[0, j, 0] + 1) / 2) * (2 ** 3 + 1)) * (2 * np.pi / (2 ** 3)))
                action_tau.append((actions_env[0, j, 1] + 1) / 2)
                actor_action.append((actions_env[0, j, 1] + 1) / 2)

        return (
            value_explore,
            action_expolit,
            action_log_probs_expolit,
            rnn_states,
            rnn_states_critic,
            actions_BS_expolit,
            action_log_probs_BS_expolit,
            action_explore,
            action_log_probs_explore,
            actions_BS_explore,
            action_log_probs_BS_explore,
            rnn_states_BS,
            action_G,
            action_phase,
            action_tau,
            value_expolit,
            self.whether_exploit
        )

    def insert(self, data):
        (obs_new, reward, dones, infos, actions, action_log_probs, rnn_states, actions_BS, action_log_probs_BS,
                rnn_states_BS, values, rnn_states_critic,  actions_exploit, actions_BS_exploit, action_log_probs_exploit,
                           action_log_probs_BS_exploit, value_preds_exploit) = data

        masks = np.ones((self.n_rollout_threads, self.n_agents, 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = np.tile(obs_new, (self.n_rollout_threads, 1, 1))
        else:
            share_obs = obs_new

        self.buffer.insert(share_obs, obs_new, rnn_states, rnn_states_BS, rnn_states_critic, actions, actions_BS, action_log_probs,
                           action_log_probs_BS, values, reward, masks, actions_exploit, actions_BS_exploit, action_log_probs_exploit,
                           action_log_probs_BS_exploit, value_preds_exploit)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.n_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                        eval_actions[:, :, i]
                    ]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.n_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.n_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.n_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.n_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")