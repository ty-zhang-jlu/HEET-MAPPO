"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
from global_mappo_8.runner.shared.base_runner import Runner
# from global_mappo_8.algorithms.algorithm.OPT_DDPG import Agent_J
from global_mappo_8.algorithms.algorithm.TD3 import TD3
from scipy.io import savemat
import csv


# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.M = 4
        self.J = 3
        self.N = 8
        self.K = 5
        self.num_bs = 1  # 基站数量
        self.B = 1  #总带宽资源
        self.n_input_j = 1
        self.n_output_j = 2 * self.J * self.K
        self.alpha = 0.00001
        self.beta = 0.0001
        self.tau = 0.005
        self.n_agents = (self.N + 1) * self.num_bs
        self.whether_exploit = False

        self.attention_weights_history = []  # 保存所有回合的注意力权重
        self.current_episode_attention = []  # 临时保存当前回合的数据

    def save_attention_data(self):
        """保存注意力权重数据用于热力图分析"""
        try:
            import pickle
            import os
            from datetime import datetime

            # 创建保存目录
            save_dir = "attention_data"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 生成文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"attention_weights_{timestamp}.pkl")

            # 保存数据
            with open(filename, 'wb') as f:
                pickle.dump({
                    'attention_history': self.attention_weights_history,
                    'num_agents': 10,  # 根据你的智能体数量调整
                    'episodes': len(self.attention_weights_history) // self.episode_length,
                    'episode_length': self.episode_length,
                    'metadata': {
                        'save_time': timestamp,
                        'total_timesteps': len(self.attention_weights_history)
                    }
                }, f)

            print(f"注意力权重数据已保存到: {filename}")
            print(f"共保存了 {len(self.attention_weights_history)} 个时间步的注意力权重")

        except Exception as e:
            print(f"保存注意力权重数据时出错: {e}")

    def save_final_attention_data(self):
        """训练结束时保存最终的注意力权重数据"""
        self.save_attention_data()

        # 同时保存为CSV格式便于查看
        try:
            import pandas as pd
            import os

            save_dir = "attention_data"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 转换为DataFrame
            data_for_csv = []
            for record in self.attention_weights_history:
                episode = record['episode']
                step = record['step']
                weights = record['weights']
                timestamp = record['timestamp']

                row = {
                    'episode': episode,
                    'step': step,
                    'timestamp': timestamp
                }

                # 添加每个智能体的注意力权重
                for i, weight in enumerate(weights):
                    row[f'agent_{i}'] = weight

                data_for_csv.append(row)

            df = pd.DataFrame(data_for_csv)
            csv_filename = os.path.join(save_dir, "attention_weights_final.csv")
            df.to_csv(csv_filename, index=False)
            print(f"注意力权重CSV文件已保存到: {csv_filename}")

        except Exception as e:
            print(f"保存CSV文件时出错: {e}")

    def run(self):
        agent_j = TD3(self.alpha, self.beta, self.n_input_j, self.n_output_j)
        # agent_j = Agent_J(self.alpha, self.beta, self.n_input_j, self.tau, self.n_output_j, gamma=0.99, max_size=10000,
        #                   C_fc1_dims=512,
        #                   C_fc2_dims=256, C_fc3_dims=128, A_fc1_dims=256, A_fc2_dims=128, batch_size=64, n_agents=1)
        # 初始化环境状态
        obs, obs_j = self.envs.envs[0].env.reset()  # 每个agent的state都是一样的，直接用这个act
        self.warmup(obs)  # 初始化环境的状态

        start = time.time()
        # print(int(self.num_env_steps), self.episode_length, self.n_rollout_threads)
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print('总迭代次数：', episodes)
        print('评估间隔：', self.eval_interval)
        reward_list = []
        rate_list = []
        Beta_t_list = []
        v_L_list = []
        Noise_list = []
        Cei_1_list = []
        Cei_2_list = []
        Cei_3_list = []
        J_Entropy_list = []
        Ers_1_list = []
        Ers_2_list = []
        Ers_3_list = []


        for episode in range(episodes):
            self.current_episode_attention = []
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)  # 学习率逐渐衰减
            done = False
            self.epsilon = 0
            # vary_L = 4 + 6 * (episode / episodes)  # 新范围：5 ~ 10
            vary_L = 10.0
            v_L_list.append(vary_L)
            # vary_noise = 0.01 + 4.99 * (episode / episodes)
            vary_noise = 0
            Noise_list.append(vary_noise)

            for step in range(self.episode_length):
                rewards = []
                rates = []
                betas = []
                # Sample actions
                (
                    value_explore,
                    rnn_states,
                    rnn_states_critic,
                    action_explore,
                    action_log_probs_explore,
                    actions_BS_explore,
                    action_log_probs_BS_explore,
                    rnn_states_BS,
                    actions_MBS_explore,
                    action_log_probs_MBS_explore,
                    rnn_states_MBS,
                    action_G,
                    action_phase,
                    action_tau,
                    action_Band,
                    self.whether_exploit,
                    entropies,
                    attention_weights,
                    attention_weights_BS,
                    attention_weights_MBS
                ) = self.collect(step)


                # print(np.asarray(obs_j).flatten())
                action_j = agent_j.choose_action(np.asarray(obs_j).flatten())
                # print("action_j NaN check:", np.isnan(action_j).any())
                w_theta = action_j[0: self.J * self.K] * 2 * np.pi
                w_beta = action_j[self.J * self.K: 2 * self.J * self.K]
                w_array = np.cos(w_theta) * w_beta + np.sin(w_theta) * w_beta * 1j
                action_W = np.reshape(w_array, (self.J, self.K))
                obs_new, glo_reward, dones, infos, obs_j_new, reward_j, sum_rate, beta_t = self.envs.envs[0].env.step(action_G, action_phase, np.array(action_tau),
                                                                                                          action_W, action_Band)
                # beta_t = 1 - 1 * (episode / episodes)
                rewards.append(glo_reward)
                rates.append(sum_rate)
                betas.append(beta_t)
                if isinstance(attention_weights, torch.Tensor):
                    # 将CUDA张量转换为CPU numpy数组
                    weights_cpu = attention_weights.cpu().detach().numpy()
                    self.current_episode_attention.append({
                        'episode': episode,
                        'step': step,
                        'weights': weights_cpu.copy(),  # 使用copy避免引用问题
                        'timestamp': episode * self.episode_length + step
                    })

                if step == self.episode_length - 1:
                    dones = [True for _ in range(self.n_agents)]
                    done = True


                data = (obs_new, glo_reward, dones, infos, action_explore, action_log_probs_explore, rnn_states, actions_BS_explore, action_log_probs_BS_explore,
                    rnn_states_BS, actions_MBS_explore, action_log_probs_MBS_explore, rnn_states_MBS, value_explore, rnn_states_critic)
                self.insert(data, entropies)
                agent_j.remember(np.asarray(obs_j).flatten(), np.asarray(action_j).flatten(),
                                 reward_j, np.asarray(obs_j_new).flatten(), done)
            if episode >= 0:
                self.compute()
                if episode <= 4000:
                    train_infos = self.train_1(vary_L, vary_noise, beta_t)
                else:
                    train_infos = self.train_2(vary_L, vary_noise)
                agent_j.learn()
                # obs_j = obs_j_new
                # obs = obs_new

                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = np.mean(rewards)
                self.log_train(train_infos, total_num_steps)
                Cei_1_list.append(train_infos['Cei_1'])
                # print("CEI_1指标：", train_infos['Cei_1'])
                Cei_2_list.append(train_infos['Cei_2'])
                # print("CEI_2指标：", train_infos['Cei_2'])
                Cei_3_list.append(train_infos['Cei_3'])
                # print("CEI_3指标：", train_infos['Cei_3'])
                J_Entropy_list.append(train_infos['J_T'])
                # print("J_T指标：", train_infos['J_T'])
                Ers_1_list.append(train_infos['Ers_1'])
                # print("ERS_1指标：", train_infos['Ers_1'])
                Ers_2_list.append(train_infos['Ers_2'])
                # print("ERS_2指标：", train_infos['Ers_2'])
                Ers_3_list.append(train_infos['Ers_3'])
                # print("ERS_3指标：", train_infos['Ers_3'])

                # 保存当前回合的完整注意力权重数据到历史记录
                if self.current_episode_attention:
                    self.attention_weights_history.extend(self.current_episode_attention)
                print(self.attention_weights_history)
                if episode % 100 == 0:  # 每100回合保存一次
                    self.save_attention_data()


            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                # print('保存一次模型')
                self.save()
                self.save_exploit()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                print("average episode rewards is ", np.mean(rewards))
                # print("average episode beta is ", np.mean(betas))
                reward_list.append(np.mean(rewards))
                rate_list.append(np.mean(rates))
                Beta_t_list.append(np.mean(betas))
                # print("average episode rates is :", np.mean(rates))

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print('评估模型一次')
                self.eval(total_num_steps)

        # 保存最终的注意力权重数据
        self.save_final_attention_data()

        file_name = 'PPO8_Reward_29.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in reward_list:
                writer.writerow([item])
        file_name = 'CEI_1_29.csv'
        with open(file_name, mode='w', newline='') as file:
               writer = csv.writer(file)
               for item in Cei_1_list:
                   writer.writerow([item])
        file_name = 'CEI_2_29.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in Cei_2_list:
                writer.writerow([item])
        file_name = 'CEI_3_29.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in Cei_3_list:
                writer.writerow([item])
        file_name = 'ERS_1_29.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in Ers_1_list:
                writer.writerow([item])
        file_name = 'ERS_2_29.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in Ers_2_list:
                writer.writerow([item])
        file_name = 'ERS_3_29.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in Ers_3_list:
                writer.writerow([item])
        print('保存数据成功RMAPPO8_29')


    def warmup(self, obs):
        # replay buffer
        if self.use_centralized_V:
            obs = np.array(obs)
            share_obs = np.repeat(obs[np.newaxis, :], self.n_rollout_threads, axis=0)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = np.repeat(share_obs[:, :6][:, np.newaxis, :], 10, axis=1)
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        self.trainer_exploit.prep_rollout()

        # target actor 的动作
        (
            value_explore,
            action_explore,
            action_log_prob_explore,
            rnn_states,
            action_BS_explore,
            action_log_probs_BS_explore,
            rnn_states_BS,
            action_MBS_explore,
            action_log_probs_MBS_explore,
            rnn_states_MBS,
            rnn_states_critic,
            dist_entropy,  # 新增返回的熵值 (n_agents,)
            dist_entropy_BS,  # 新增返回的熵值 (1,)
            dist_entropy_MBS,  # 新增返回的熵值 (1,)
            attention_weights,
            attention_weights_BS,
            attention_weights_MBS
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # 将熵值组合成数组，形状为 (n_rollout_threads, n_agents, 1)
        # 假设 dist_entropy 对应前8个智能体，dist_entropy_BS 对应第9个智能体，dist_entropy_MBS 对应第10个智能体
        # 注意：根据您的智能体数量调整
        n_agents = self.n_agents
        entropies = np.zeros((1, n_agents, 1), dtype=np.float32)
        entropies[0, :7, 0] = dist_entropy.mean().item()  # 前8个智能体
        entropies[0, 8, 0] = dist_entropy_BS.mean().item()  # 第9个智能体
        entropies[0, n_agents-1, 0] = dist_entropy_MBS.mean().item()  # 第10个智能体（如果存在）

        value = value_explore
        action = action_explore
        action_BS = action_BS_explore
        action_MBS = action_MBS_explore

        # 分割为各个线程
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        rnn_states_BS = np.array(
            np.split(_t2n(rnn_states_BS), self.n_rollout_threads))
        rnn_states_MBS = np.array(
            np.split(_t2n(rnn_states_MBS), self.n_rollout_threads))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        value_explore = np.array(np.split(_t2n(value_explore), self.n_rollout_threads))

        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        actions_BS = np.array(np.split(_t2n(action_BS), self.n_rollout_threads))
        actions_MBS = np.array(np.split(_t2n(action_MBS), self.n_rollout_threads))

        action_explore = np.array(np.split(_t2n(action_explore), self.n_rollout_threads))
        action_log_probs_explore = np.array(
            np.split(_t2n(action_log_prob_explore), self.n_rollout_threads)
        )
        actions_BS_explore = np.array(np.split(_t2n(action_BS_explore), self.n_rollout_threads))
        action_log_probs_BS_explore = np.array(
            np.split(_t2n(action_log_probs_BS_explore), self.n_rollout_threads)
        )
        actions_MBS_explore = np.array(np.split(_t2n(action_MBS_explore), self.n_rollout_threads))
        action_log_probs_MBS_explore = np.array(
            np.split(_t2n(action_log_probs_MBS_explore), self.n_rollout_threads)
        )

        # 重构动作处理逻辑以支持多BS场景
        all_action_G = []
        all_action_phase = []
        all_action_tau = []
        all_actor_action = []
        all_action_Mbs = []

        # 假设每个BS有固定数量的agent (n_agents_per_bs)
        n_agents_per_bs = self.n_agents // self.num_bs

        for bs_idx in range(self.num_bs):
            action_G = None
            action_phase = []
            action_tau = []
            actor_action = []

            # 处理当前BS对应的agent动作
            start_idx = bs_idx * n_agents_per_bs
            end_idx = (bs_idx + 1) * n_agents_per_bs

            for j in range(start_idx, end_idx):
                if j == end_idx - 1:  # 每个BS组的最后一个agent处理BS动作
                    # 处理BS动作
                    action_BS_clipped = np.clip(actions_BS[0, j, :], -0.999, 0.999).reshape(-1)
                    g_theta = action_BS_clipped[0: self.M * self.K] * np.pi
                    actor_action.extend(g_theta.tolist())
                    g_beta = (action_BS_clipped[self.M * self.K: 2 * self.M * self.K] + 1) / 2
                    actor_action.extend(g_beta.tolist())
                    g_array = np.cos(g_theta) * g_beta + np.sin(g_theta) * g_beta * 1j
                    action_G = np.reshape(g_array, (self.M, self.K))
                else:
                    # 处理RIS相位和tau动作
                    phase = np.floor(((actions[0, j, 0] + 1) / 2) * (2 ** 3 + 1)) * (2 * np.pi / (2 ** 3))
                    action_phase.append(phase)
                    actor_action.append(phase)

                    tau = (actions[0, j, 1] + 1) / 2
                    action_tau.append(tau)
                    actor_action.append(tau)

            all_action_G.append(action_G)
            all_action_phase.append(action_phase)
            all_action_tau.append(action_tau)
            all_actor_action.append(actor_action)

        action_Band = self.B * (action_MBS[0] + 1) / torch.sum((action_MBS[0] + 1))

        return (
            value_explore,
            rnn_states,
            rnn_states_critic,
            action_explore,
            action_log_probs_explore,
            actions_BS_explore,
            action_log_probs_BS_explore,
            rnn_states_BS,
            actions_MBS_explore,
            action_log_probs_MBS_explore,
            rnn_states_MBS,
            all_action_G,  # 改为返回所有BS的G动作列表
            all_action_phase,  # 改为返回所有BS的phase动作列表
            all_action_tau,  # 改为返回所有BS的tau动作列表
            action_Band,
            self.whether_exploit,
            entropies,
            attention_weights,
            attention_weights_BS,
            attention_weights_MBS
        )

    def insert(self, data, entropies):
        (obs_new, reward, dones, infos, actions, action_log_probs, rnn_states, actions_BS, action_log_probs_BS,
                rnn_states_BS, actions_MBS, action_log_probs_MBS, rnn_states_MBS, values, rnn_states_critic) = data

        masks = np.ones((self.n_rollout_threads, self.n_agents, 1), dtype=np.float32)
        # masks[dones == True] = np.zeros((sum(dones), 1), dtype=np.float32)


        if self.use_centralized_V:
            # share_obs = obs_new.reshape(self.n_rollout_threads, -1)
            share_obs = np.tile(obs_new, (self.n_rollout_threads, 1, 1))  # 直接复制32次
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        else:
            share_obs = obs_new

        self.buffer.insert(share_obs, obs_new, rnn_states, rnn_states_BS, rnn_states_MBS, rnn_states_critic, actions, actions_BS, actions_MBS, action_log_probs,
                           action_log_probs_BS, action_log_probs_MBS, values, reward, masks, entropies=entropies)

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

            # Obser reward and next obs
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
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
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

                # Obser reward and next obs
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

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
