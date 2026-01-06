import numpy as np
import math
from scipy.stats import gaussian_kde
import random
import torch


class EnvCore(object):
    """
    # 环境中的智能体 - 多基站多Jammer版本
    """

    def __init__(self):
        self.M = 4  # 每个BS的天线数量
        self.N = 16  # 每个RIS的元件数量
        self.K = 5  # 每个BS服务的用户数量
        self.Z = 5  # 障碍物数量
        self.c1 = 11.95  # 环境常数1
        self.c2 = 0.136  # 环境常数2
        self.kappa = 0.15  # 障碍物密度
        self.dR = 0.1  # RIS元件间距
        self.dB = 0.2  # BS天线间距
        self.lambda_c = 35 / 3  # 载波波长
        self.P_tx = 500  # 发射功率
        self.sigma_noise = 10 ** (-13)  # 噪声功率谱密度
        self.RIS_height = 400  # RIS高度
        self.Sigma2BR = 0.01
        self.Sigma2RU = 0.01
        self.K_BR = 3.5
        self.K_RU = 2.2
        self.Sigma_csi = 0.02
        self.beta_min = 0.2
        self.theta_bar = 0.1
        self.kappa_bar = 0.4
        self.db_min = 2
        self.pin_min = 25
        self.yita = 0.9
        self.P_Jammer = 0.5  # Jammer发射功率约束
        self.J = 3  # Jammer天线数量
        self.Sigma_csi_Jammer = 0.02  # Jammer处CSI error
        self.dJ = 0.2  # Jammer天线间距

        # 新增参数
        self.num_bs = 2  # 基站数量
        self.B = 3 * self.num_bs  # 总带宽(MHz)

        self.agent_num = (self.N + 1) * self.num_bs  # 设置智能体的个数(每个BS-RIS组)
        self.obs_dim = self.num_bs * 2  # 设置智能体的观测维度
        self.action_dim = [2, 2 * self.M * self.K, self.num_bs]  # 设置智能体的动作维度
        self.done = False
        self.update = True

        # 初始化多个基站、Jammer和用户位置
        self.BS_positions = []
        self.Jammer_positions = []
        self.user_positions_list = []
        self.RIS_center_positions = []
        self.RIS_element_positions_list = []

        for bs_idx in range(self.num_bs):
            # 基站位置 - 在500x500区域内均匀分布
            bs_pos = np.array([500 * (bs_idx % 3), 500 * (bs_idx // 3), 10])
            self.BS_positions.append(bs_pos)

            # Jammer位置 - 在基站附近随机分布，高度与RIS相同
            jammer_pos = bs_pos + np.array(
                [random.uniform(1000, 200), random.uniform(1000, 200), self.RIS_height - bs_pos[2]])
            self.Jammer_positions.append(jammer_pos)

            # 用户位置 - 围绕各自基站随机分布
            user_positions = np.random.rand(self.K, 3) * np.array([[200, 200, 2 - 0.5]]) + \
                             np.array([bs_pos[0] - 100, bs_pos[1] - 100, 0.5])
            self.user_positions_list.append(user_positions)

            # RIS位置 - 基于用户分布确定
            ris_center = self.determine_ris_position_via_kde(user_positions)
            self.RIS_center_positions.append(ris_center)

            # RIS元件位置
            random_offsets = np.arange(self.N)[:, np.newaxis] * self.dR * (np.random.rand(self.N, 2) - 0.5)
            ris_element_positions = ris_center + random_offsets
            ris_element_positions = np.hstack((ris_element_positions, np.full((self.N, 1), self.RIS_height)))
            self.RIS_element_positions_list.append(ris_element_positions)

        # 初始化障碍物位置 (共享的障碍物环境)
        self.obstacle_positions = np.random.rand(self.Z, 3) * np.array([[1500, 1500, 0]]) + np.array(
            [[0, 0, 0]])  # 障碍物随机分布在1500x1500的区域内，高度在1.25-5m
        self.obstacle_sizes = np.random.rand(self.Z, 3) * np.array([[5, 5, 3.75]]) + np.array(
            [[5, 5, 1.25]])  # 障碍物尺寸随机分布在1-2m长和宽，1.25-5m高
        self.ave_l = np.mean(self.obstacle_sizes[:, 0])
        self.ave_w = np.mean(self.obstacle_sizes[:, 1])
        self.ave_h = np.mean(self.obstacle_sizes[:, 2])

    def reset(self):
        state_old_all = []
        state_old_all_j = []

        for bs_idx in range(self.num_bs):
            sum_rate_0 = random.uniform(0, 50)
            sum_power_in_0 = random.uniform(0, 40)
            rate_min_0 = np.log2(1 + 10 ** (6 / 10))
            rate_list_0 = [random.uniform(0, 5) for _ in range(self.K)]
            h_k_list_0 = [random.uniform(0, 1) for _ in range(self.K)]

            sub_obs = self.get_state(sum_power_in_0, h_k_list_0, bs_idx)  # 每个agent的观察空间一致
            state_old_all.append(sub_obs)
            state_old_all_flattened = np.concatenate(state_old_all)

            # Jammer的状态只包含对应BS的信息
            state_old_all_j.append(self.get_state_for_jammer(1 / sum_rate_0, bs_idx))

        return state_old_all_flattened, state_old_all_j[0]

    def step(self, actions_G, actions_phase, actions_tau, actions_W, actions_bandwidth):
        # actions_bandwidth: 每个基站分配的带宽列表，形状为(num_bs,)
        self.update = True

        if self.done:
            sub_agent_done = [True for _ in range(self.agent_num * self.num_bs)]
            sub_agent_obs, j_agent_obs = self.reset()
            sub_agent_reward = [[0.] for _ in range(self.agent_num * self.num_bs)]
            j_agent_reward = 0.
            sub_agent_info = [{} for _ in range(self.agent_num * self.num_bs)]
            sub_sum_rate = [0. for _ in range(self.num_bs)]
            global_reward = 0.
            Beta_t = 0.

            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info,
                    j_agent_obs, j_agent_reward, sub_sum_rate, global_reward, Beta_t]

        global_reward = 0
        all_sub_agent_reward = []
        all_sub_agent_obs = []
        all_j_agent_reward = []
        all_j_agent_obs = []
        all_sub_sum_rate = []
        all_sum_power_in = []
        all_rate_list = []


        for bs_idx in range(self.num_bs):
            # 获取当前基站对应的动作
            action_G = actions_G[bs_idx]
            action_phase = actions_phase[bs_idx]
            action_tau = actions_tau[bs_idx]
            action_W = actions_W
            bandwidth = actions_bandwidth[bs_idx]  # 当前基站分配的带宽

            # 计算奖励和状态
            reward, next_state, j_reward, next_state_j, sum_rate, sum_power_in, rate_list = self.calculate_reward(
                action_G, action_phase, action_tau, action_W, bs_idx, bandwidth)

            global_reward += reward  # 累加所有基站的总速率
            all_sum_power_in.append(sum_power_in)
            all_rate_list.append(rate_list)

            all_sub_agent_reward.extend([reward])  # 每个基站有agent_num个智能体
            all_sub_agent_obs.extend([next_state])
            all_j_agent_reward.append(j_reward)
            all_j_agent_obs.append(next_state_j)
            all_sub_sum_rate.append(sum_rate)
        all_sub_agent_obs = np.sum(all_sub_agent_obs)
        # print("全局奖励：", global_reward)

        sub_agent_done = [False for _ in range(self.agent_num * self.num_bs)]
        sub_agent_info = [{} for _ in range(self.agent_num * self.num_bs)]

        sigma = 0.1  # 可调整为类参数或从外部传入

        # 收集所有用户的截断速率（式35）
        all_truncated_rates = []
        for bs_power in all_sum_power_in:
            if bs_power < self.pin_min:
                # 只要有一个基站不满足条件，就设置标志为 False 并退出循环
                self.update = False
                # print(self.update)
                break

        beta_t = 1
        # print("用户实际速率", all_rate_list)
        # if self.update:
            # print("用户历史速率最大值", self.max_historical_rates)

        # print("用户截断值", all_truncated_rates)

        # print('average_beta_t', beta_t)

        return [all_sub_agent_obs, global_reward, sub_agent_done, sub_agent_info,
                np.sum(all_j_agent_obs), np.sum(all_j_agent_reward), all_sub_sum_rate, beta_t]

    def determine_ris_position_via_kde(self, user_positions):
        # 提取用户的x和y坐标
        users_xy = user_positions[:, :2]
        centroid = users_xy.mean(axis=0)
        return centroid

    def calculate_LoS_probability(self, d, elevation_angle):
        xi = 180 / math.pi * math.asin(elevation_angle / math.sqrt(d ** 2 + elevation_angle ** 2))
        return 1 / (1 + self.c1 * math.exp(-self.c2 * (xi - self.c1)))

    def calculate_LoS_RU(self, zRn, zk, d):
        Plos = (1 + (zRn - self.ave_h) / zk) / 2 * np.exp(-(2 * self.kappa * (self.ave_l + self.ave_w)) * d / math.pi
                                                          + self.kappa * self.ave_w * self.ave_l)
        return Plos

    def calculate_path_loss(self, d, LoS_prob):
        alpha = 3
        theta = 100
        PL_n = (LoS_prob + (1 - LoS_prob) * theta) * (d ** (-alpha))
        return PL_n

    def calculate_Jammer_RIS(self, ris_element_positions, jammer_position):
        H_LoS_JR = np.zeros((self.J, self.N), dtype=complex)
        H_NLoS_JR = np.zeros((self.J, self.N), dtype=complex)
        H_CSI_JR = np.zeros((self.J, self.N), dtype=complex)
        for x in range(self.J):
            for n in range(self.N):
                d = np.linalg.norm(jammer_position - ris_element_positions[n])
                PL_JR = 10 ** (-30 / 10) * (d ** (-3.5))
                phi_JR_D = (ris_element_positions[n, 0] - jammer_position[0]) / d
                phi_JR_A = (jammer_position[0] - ris_element_positions[n, 0]) / d
                aR_n = np.exp(-1j * 2 * math.pi / self.lambda_c * (n - 1) * self.dR * phi_JR_D)
                aB_x = np.exp(-1j * 2 * math.pi / self.lambda_c * (x - 1) * self.dJ * phi_JR_A)
                H_LoS_JR[x, n] = np.sqrt(PL_JR) * np.dot(aB_x, aR_n.conj().T)
                H_NLoS_JR[x, n] = np.sqrt(PL_JR) * ((np.random.randn(1) + 1j * np.random.randn(1))
                                                    / np.sqrt(2) * np.sqrt(self.Sigma2BR))

        H_CSI_JR = (np.random.randn(self.J, self.N) + 1j * np.random.randn(self.J, self.N)) / np.sqrt(2) * np.sqrt(
            self.Sigma_csi_Jammer)
        H_JR = (np.sqrt(self.K_BR) / np.sqrt(self.K_BR + 1)) * H_LoS_JR + (1 / np.sqrt(self.K_BR + 1)) * H_NLoS_JR
        return H_JR

    def calculate_channel_matrix(self, bs_idx):
        H_LoS_BR = np.zeros((self.M, self.N), dtype=complex)
        H_NLoS_BR = np.zeros((self.M, self.N), dtype=complex)
        H_BR = np.zeros((self.M, self.N), dtype=complex)
        for m in range(self.M):
            for n in range(self.N):
                d = np.linalg.norm(self.BS_positions[bs_idx] - self.RIS_element_positions_list[bs_idx][n])
                elevation_angle = (self.RIS_element_positions_list[bs_idx][n, 2] - self.BS_positions[bs_idx][2]) / d
                LoS_prob = self.calculate_LoS_probability(d, elevation_angle)
                PL = self.calculate_path_loss(d, LoS_prob)
                phi_BR_D = (self.RIS_element_positions_list[bs_idx][n, 0] - self.BS_positions[bs_idx][0]) / d
                phi_BR_A = (self.BS_positions[bs_idx][0] - self.RIS_element_positions_list[bs_idx][n, 0]) / d
                aR_n = np.exp(-1j * 2 * math.pi / self.lambda_c * (n - 1) * self.dR * phi_BR_D)
                aB_m = np.exp(-1j * 2 * math.pi / self.lambda_c * (m - 1) * self.dB * phi_BR_A)
                H_LoS_BR[m, n] = np.sqrt(PL) * np.dot(aB_m, aR_n.conj().T)
                H_NLoS_BR[m, n] = np.sqrt(PL) * (
                            (np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2) * np.sqrt(self.Sigma2BR))
        H_BR = (np.sqrt(self.K_BR) / np.sqrt(self.K_BR + 1)) * H_LoS_BR + (1 / np.sqrt(self.K_BR + 1)) * H_NLoS_BR
        return H_BR

    def calculate_channel_matrix_RU(self, bs_idx):
        H_LoS_RU = np.zeros((self.N, self.K), dtype=complex)
        H_NLoS_RU = np.zeros((self.N, self.K), dtype=complex)
        H_CSI_RU = np.zeros((self.N, self.K), dtype=complex)
        H_RU = np.zeros((self.N, self.K), dtype=complex)
        for n in range(self.N):
            for k in range(self.K):
                d = np.sqrt(
                    (self.RIS_element_positions_list[bs_idx][n, 0] - self.user_positions_list[bs_idx][k, 0]) ** 2 +
                    (self.RIS_element_positions_list[bs_idx][n, 1] - self.user_positions_list[bs_idx][k, 1]) ** 2 +
                    (self.RIS_element_positions_list[bs_idx][n, 2] - self.user_positions_list[bs_idx][k, 2]) ** 2)
                LoS_prob = self.calculate_LoS_RU(self.RIS_element_positions_list[bs_idx][n, 2],
                                                 self.user_positions_list[bs_idx][k, 2], d)
                PL = self.calculate_path_loss(d, LoS_prob)
                aR_n = np.exp(-1j * 2 * math.pi / self.lambda_c * (n - 1) * self.dR *
                              (self.RIS_element_positions_list[bs_idx][n, 0] - self.user_positions_list[bs_idx][
                                  k, 0]) / d)
                H_LoS_RU[n, k] = np.sqrt(PL) * aR_n
                H_NLoS_RU[n, k] = np.sqrt(PL) * (
                            (np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2) * np.sqrt(self.Sigma2RU))
        H_CSI_RU = (np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K)) / np.sqrt(2) * np.sqrt(
            self.Sigma_csi)
        H_RU = (1 / np.sqrt(self.K_RU + 1)) * (
                    np.sqrt(self.K_RU) * H_LoS_RU + (1 / np.sqrt(self.K_RU + 1)) * H_NLoS_RU) + H_CSI_RU
        return H_RU

    def PDA_model(self, angles):
        betas = (1 - self.beta_min) * ((np.sin(angles - self.theta_bar) + 1) / 2) ** self.kappa_bar + self.beta_min
        return betas

    def calculate_phase_shift_matrix(self, action_phase):
        Beta = np.zeros(self.N)
        PDA = np.zeros(self.N, dtype=complex)
        for n_ris in range(self.N):
            Beta[n_ris] = self.PDA_model(action_phase[n_ris])
            PDA[n_ris] = Beta[n_ris] * np.exp(1j * action_phase[n_ris])
        Theta = np.diag(PDA)
        return Theta

    def calculate_sinr(self, G, W, user_index, tau, Theta, H_BR, H_RU, H_JR, bandwidth):
        sum_power_j = 0
        Tau_Matrix = np.diag(tau)
        IT_Matrix = np.diag(1 - tau)
        power_t_1 = np.sum(np.abs(np.dot((H_BR @ Tau_Matrix).conj().T, (G)))) * self.P_tx
        power_t_2 = np.sum(np.abs(np.dot((H_JR @ Tau_Matrix).conj().T, (W)))) * self.P_Jammer
        sum_power_t = power_t_1 + power_t_2
        # print("H_BR NaN check:", np.isnan(H_BR).any())
        # print("Tau_Matrix NaN check:", np.isnan(Tau_Matrix).any())
        # print("G NaN check:", np.isnan(G).any())
        # print("W NaN check:", np.isnan(W).any())
        h_k = H_RU[:, user_index].reshape(1, -1) @ Theta @ IT_Matrix @ H_BR.conj().T @ (G[:, user_index])
        signal_power = np.abs(h_k * self.P_tx) ** 2

        for j_r in range(self.J):
            h_j = H_RU[:, j_r].reshape(1, -1) @ Theta @ IT_Matrix @ H_JR.conj().T @ (W[:, j_r])
            power_j = np.abs(h_j * self.P_Jammer) ** 2
            sum_power_j = sum_power_j + power_j

        interference_power = self.P_tx * sum(
            np.abs(H_RU[:, j].reshape(1, -1) @ Theta @ IT_Matrix @ H_BR.conj().T @ G[:, j]) ** 2
            for j in range(self.K) if j != user_index)
        sinr_db = 10 * np.log10(signal_power / (sum_power_j + self.sigma_noise))
        sinr_linear = 10 ** (sinr_db / 10)
        if torch.is_tensor(sinr_linear):
            # 确保张量在CPU上，然后再转换为numpy
            sinr_linear = sinr_linear.detach().cpu()  # 使用detach()确保梯度不会传播
            sinr_linear = sinr_linear.numpy()
        if torch.is_tensor(bandwidth):
            # 确保张量在CPU上，然后再转换为numpy
            bandwidth = bandwidth.detach().cpu()  # 使用detach()确保梯度不会传播
            bandwidth = bandwidth.numpy()
        if bandwidth == 0:
            bandwidth = 0.000001
        rate = bandwidth * np.log2(1 + sinr_linear)  # 考虑带宽影响
        return sinr_db, sum_power_t, rate, signal_power

    def calculate_reward(self, action_G, action_phase, action_tau, action_W, bs_idx, bandwidth):
        rate_list = []
        h_k_list = []
        if torch.is_tensor(bandwidth):
            # 确保张量在CPU上，然后再转换为numpy
            bandwidth = bandwidth.detach().cpu()  # 使用detach()确保梯度不会传播
            bandwidth = bandwidth.numpy()
        if bandwidth == 0:
            bandwidth = 0.000001
        rate_min = bandwidth * np.log2(1 + 10 ** (self.db_min / 10))
        sum_rate = 0
        sum_rate_for_jammer = 0
        sum_power_t = 0
        reward = 0
        reward_1 = 0  # 有关rate的惩罚，负数
        reward_2 = 0  # 有关EH的惩罚，负数
        U_sinr = np.zeros(self.K)
        Theta = self.calculate_phase_shift_matrix(action_phase)
        H_BR = self.calculate_channel_matrix(bs_idx)
        H_JR = self.calculate_Jammer_RIS(self.RIS_element_positions_list[bs_idx], self.Jammer_positions[bs_idx])
        H_RU = self.calculate_channel_matrix_RU(bs_idx)

        for u in range(self.K):
            U_sinr[u], sum_power_t, u_rate, signal_power = self.calculate_sinr(
                action_G, action_W, u, action_tau, Theta, H_BR, H_RU, H_JR, bandwidth)
            rate_list.append(u_rate)  # 存放每个用户的速率
            h_k_list.append(signal_power)
            if u_rate < rate_min:
                reward_1 = reward_1 + u_rate - rate_min
            sum_rate = sum_rate + u_rate

        reward = reward + min(rate_list)  # 最小速率奖励
        sum_power_in = self.yita * sum_power_t
        # print("每个RIS的吸收能量：", sum_power_in)
        # print("用户速率最大值：", max(rate_list))
        if sum_power_in < self.pin_min:
            reward_2 = reward_2 + sum_power_in - self.pin_min
            # print("能量惩罚：", reward_2)

        if sum_power_in < self.pin_min:
            # print("奖励", reward, "速率惩罚", reward_1, "能量惩罚", reward_2)
            reward = 0.4 * reward + 0.30 * reward_1 + 0.30 * reward_2
            # print("最终奖励", reward)

        next_state = self.get_state(sum_power_in, h_k_list, bs_idx)
        next_state_for_jammer = self.get_state_for_jammer(sum(rate_list), bs_idx)

        return reward, next_state, 1 / max(rate_list), next_state_for_jammer, sum_rate, sum_power_in, rate_list

    def get_state(self, sum_power_in, h_k_list, bs_idx):
        state_list = []
        state_sum_power_in = np.array(sum_power_in, ndmin=1)
        state_h_k_list = np.array(h_k_list).flatten()
        state_P_in = np.array(self.pin_min, ndmin=1)

        # 添加基站位置信息
        # state_bs_pos = np.array(self.BS_positions[bs_idx][:2], ndmin=1)  # 只取x,y坐标

        # 添加Jammer位置信息
        # state_jammer_pos = np.array(self.Jammer_positions[bs_idx][:2], ndmin=1)

        state_list.append(state_sum_power_in)
        # state_list.append(state_h_k_list)
        state_list.append(state_P_in)
        # state_list.append(state_bs_pos)
        # state_list.append(state_jammer_pos)

        state_array = np.concatenate(state_list)
        return state_array

    def get_state_for_jammer(self, sum_rate_for_jammer, bs_idx):
        state_list_jammer = []
        state_sum_rate_for_jammer = np.array(sum_rate_for_jammer, ndmin=1)

        # 添加Jammer对应的BS位置信息
        state_bs_pos = np.array(self.BS_positions[bs_idx][:2], ndmin=1)

        # 添加Jammer自身位置信息
        state_jammer_pos = np.array(self.Jammer_positions[bs_idx][:2], ndmin=1)

        state_list_jammer.append(state_sum_rate_for_jammer)
        # state_list_jammer.append(state_bs_pos)
        # state_list_jammer.append(state_jammer_pos)

        state_array_jammer = np.concatenate(state_list_jammer)
        return state_array_jammer