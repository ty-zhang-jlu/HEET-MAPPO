import numpy as np
import math
from scipy.stats import gaussian_kde
import random


class EnvCore(object):
    def __init__(self):
        self.M = 4
        self.N = 16
        self.K = 5
        self.Z = 10
        self.c1 = 9.61
        self.c2 = 0.16
        self.kappa = 0.15
        self.lambda_c = 35/3
        self.dR = self.lambda_c/2
        self.dB = self.lambda_c/2
        self.dJ = self.lambda_c/2
        self.P_tx = 100
        self.sigma_noise = 10**(-10.2)
        self.RIS_height = 400
        self.Sigma2BR = 0.01
        self.Sigma2RU = 0.01
        self.K_BR = 2
        self.K_RU = 2
        self.Sigma_csi = 0.01
        self.beta_min = 0.2
        self.theta_bar = 0.1
        self.kappa_bar = 0.4
        self.db_min = 14.9
        self.pin_min = 40
        self.yita = 0.7
        self.Jammer_position = [400, 400, 400]
        self.P_Jammer = 0.5
        self.J = 5
        self.Sigma_csi_Jammer = 0.01
        self.L0 = 10**(-30/10)
        self.alpha = 2.2
        self.theta = 100

        self.agent_num = self.N + 1
        self.obs_dim = self.K + 2
        self.action_dim = [2, 2 * self.M * self.K]
        self.done = False

        self.BS_position = np.array([0, 0, 10])
        self.user_positions = np.random.rand(self.K, 3) * np.array([[500, 500, 2 - 0.5]]) + np.array([[0, 0, 0.5]])
        self.RIS_center_position = self.determine_ris_position_via_kde()
        random_offsets = np.arange(self.N)[:, np.newaxis] * self.dR * (np.random.rand(self.N, 2) - 0.5)
        self.RIS_element_positions = self.RIS_center_position + random_offsets
        self.RIS_element_positions = np.hstack((self.RIS_element_positions, np.full((self.N, 1), self.RIS_height)))
        self.obstacle_positions = np.random.rand(self.Z, 3) * np.array([[500, 500, 0]]) + np.array([[0, 0, 0]])
        self.obstacle_sizes = np.random.rand(self.Z, 3) * np.array([[5, 5, 3.75]]) + np.array([[5, 5, 1.25]])
        self.ave_l = np.mean(self.obstacle_sizes[:, 0])
        self.ave_w = np.mean(self.obstacle_sizes[:, 1])
        self.ave_h = np.mean(self.obstacle_sizes[:, 2])


    def reset(self):
        state_old_all = []

        sum_rate_0 = random.uniform(0, 50)
        sum_power_in_0 = random.uniform(0, 80)
        rate_min_0 = np.log2(1 + 10 ** (self.db_min / 10))
        rate_list_0 = [random.uniform(0, 5) for _ in range(self.K)]
        h_k_list_0 = [random.uniform(0, 1) for _ in range(self.K)]

        for i in range(self.agent_num):
            sub_obs = self.get_state(sum_power_in_0, h_k_list_0)
            state_old_all.append(sub_obs)

        state_old_all_j = self.get_state_for_jammer(1 / sum_rate_0)
        return state_old_all, state_old_all_j

    def step(self, action_G, action_phase,action_tau, action_W):
        if self.done:
            sub_agent_done = [True for _ in range(self.agent_num)]
            sub_agent_obs, j_agent_obs = self.reset()
            sub_agent_reward = [[0.] for _ in range(self.agent_num)]
            j_agent_reward = [0.]
            sub_agent_info = [{} for _ in range(self.agent_num)]
            sub_sum_rate = [0.]

            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, j_agent_obs, j_agent_reward, sub_sum_rate]


        sub_agent_reward, sub_agent_obs, j_agent_reward, j_agent_obs, sub_sum_rate, sub_agent_fair, sub_rate, sub_power = self.calculate_reward(action_G, action_phase,
                                                                                                             action_tau, action_W)
        sub_agent_done = [False for _ in range(self.agent_num)]
        sub_agent_info = [{} for _ in range(self.agent_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, j_agent_obs, j_agent_reward, sub_sum_rate, sub_agent_fair, sub_rate, sub_power]


    def determine_ris_position_via_kde(self):
        users_xy = self.user_positions[:, :2]
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
        PL_n = (LoS_prob + (1 - LoS_prob) * self.theta) * self.L0 * (d ** (-self.alpha))
        return PL_n

    def calculate_Jammer_RIS(self):
        H_LoS_JR = np.zeros((self.J, self.N), dtype=complex)
        H_NLoS_JR = np.zeros((self.J, self.N), dtype=complex)
        H_CSI_JR = np.zeros((self.J, self.N), dtype=complex)
        for x in range(self.J):
            for n in range(self.N):
                d = np.linalg.norm(self.Jammer_position - self.RIS_element_positions[n])
                PL_JR = self.L0 * (d ** (-self.alpha))
                phi_JR_D = (self.RIS_element_positions[n, 0] - self.Jammer_position[0]) / d
                phi_JR_A = (self.Jammer_position[0] - self.RIS_element_positions[n, 0]) / d
                aR_n = np.exp(-1j * 2 * math.pi / self.lambda_c * (n - 1) * self.dR * phi_JR_D)
                aB_x = np.exp(-1j * 2 * math.pi / self.lambda_c * (x - 1) * self.dJ * phi_JR_A)
                H_LoS_JR[x, n] = np.sqrt(PL_JR) * np.dot(aB_x, aR_n.conj().T)
                H_NLoS_JR[x, n] = np.sqrt(PL_JR) * ((np.random.randn(1) + 1j * np.random.randn(1))
                                                    / np.sqrt(2) * np.sqrt(self.Sigma2BR))

        H_CSI_JR = (np.random.randn(self.J, self.N) + 1j * np.random.randn(self.J, self.N)) / np.sqrt(2) * np.sqrt(self.Sigma_csi_Jammer)
        H_JR = (np.sqrt(self.K_BR) / np.sqrt(self.K_BR + 1)) * H_LoS_JR + (1 / np.sqrt(self.K_BR + 1)) * H_NLoS_JR
        return H_JR

    def calculate_channel_matrix(self):
        H_LoS_BR = np.zeros((self.M, self.N), dtype=complex)
        H_NLoS_BR = np.zeros((self.M, self.N), dtype=complex)
        H_BR = np.zeros((self.M, self.N), dtype=complex)
        for m in range(self.M):
            for n in range(self.N):
                d = np.linalg.norm(self.BS_position - self.RIS_element_positions[n])
                elevation_angle = (self.RIS_element_positions[n, 2] - self.BS_position[2]) / d
                LoS_prob = self.calculate_LoS_probability(d, elevation_angle)
                PL = self.calculate_path_loss(d, LoS_prob)
                phi_BR_D = (self.RIS_element_positions[n, 0] - self.BS_position[0]) / d
                phi_BR_A = (self.BS_position[0] - self.RIS_element_positions[n, 0]) / d
                aR_n = np.exp(-1j * 2 * math.pi / self.lambda_c * (n - 1) * self.dR * phi_BR_D)
                aB_m = np.exp(-1j * 2 * math.pi / self.lambda_c * (m - 1) * self.dB * phi_BR_A)
                H_LoS_BR[m, n] = np.sqrt(PL) * np.dot(aB_m, aR_n.conj().T)
                H_NLoS_BR[m, n] = np.sqrt(PL) *((np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2) * np.sqrt(self.Sigma2BR))
        H_BR = (np.sqrt(self.K_BR) / np.sqrt(self.K_BR + 1)) * H_LoS_BR + (1 / np.sqrt(self.K_BR + 1)) * H_NLoS_BR
        return H_BR

    def calculate_channel_matrix_RU(self):
        H_LoS_RU = np.zeros((self.N, self.K), dtype=complex)
        H_NLoS_RU = np.zeros((self.N, self.K), dtype=complex)
        H_CSI_RU = np.zeros((self.N, self.K), dtype=complex)
        H_RU = np.zeros((self.N, self.K), dtype=complex)
        for n in range(self.N):
            for k in range(self.K):
                d = np.sqrt((self.RIS_element_positions[n, 0] - self.user_positions[k, 0]) ** 2 + (
                            self.RIS_element_positions[n, 1]
                            - self.user_positions[k, 1]) ** 2 + (
                                               self.RIS_element_positions[n, 2] - self.user_positions[k, 2]) ** 2)
                LoS_prob = self.calculate_LoS_RU(self.RIS_element_positions[n, 2], self.user_positions[k, 2], d)
                PL = self.calculate_path_loss(d, LoS_prob)
                aR_n = np.exp(-1j * 2 * math.pi / self.lambda_c * (n - 1) * self.dR * (self.RIS_element_positions[n, 0] - self.user_positions[k, 0]) / d)
                H_LoS_RU[n, k] = np.sqrt(PL) * aR_n
                H_NLoS_RU[n, k] = np.sqrt(PL) *((np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2) * np.sqrt(self.Sigma2RU))
        H_CSI_RU = (np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K)) / np.sqrt(2) * np.sqrt(self.Sigma_csi)
        H_RU = (1 / np.sqrt(self.K_RU + 1)) * (np.sqrt(self.K_RU) * H_LoS_RU + (1 / np.sqrt(self.K_RU + 1)) * H_NLoS_RU) + H_CSI_RU

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

    def calculate_sinr(self, G, W, user_index, tau, Theta, H_BR, H_RU, H_JR):
        sum_power_j = 0
        Tau_Matrix = np.diag(tau)
        IT_Matrix = np.diag(1-tau)
        power_t_1 = np.sum(np.abs(np.dot((H_BR @ Tau_Matrix).conj().T, (G)))) * self.P_tx
        power_t_2 = np.sum(np.abs(np.dot((H_JR @ Tau_Matrix).conj().T, (W)))) * self.P_Jammer
        sum_power_t = power_t_1 + power_t_2
        h_k = H_RU[:, user_index].reshape(1, -1) @ Theta @ IT_Matrix @ H_BR.conj().T @ (G[:, user_index])
        signal_power = np.abs(h_k * self.P_tx) ** 2

        for j_r in range(self.J):
            h_j = H_RU[:, j_r].reshape(1, -1) @ Theta @ IT_Matrix @ H_JR.conj().T @ (W[:, j_r])
            w_norm = np.linalg.norm(W[:, j_r])
            power_j = np.abs(h_j * self.P_Jammer) ** 2
            sum_power_j = sum_power_j + power_j

        interference_power = self.P_tx * sum(np.abs(H_RU[:, j].reshape(1, -1) @ Theta @IT_Matrix @ H_BR.conj().T @ G[:, j]) ** 2 for j in range(self.K) if j != user_index)
        sinr_db = 10 * np.log10(signal_power / (sum_power_j + self.sigma_noise))
        sinr_linear = 10 ** (sinr_db / 10)
        rate = np.log2(1 + sinr_linear)
        return sinr_db, sum_power_t, rate, signal_power

    def calculate_reward(self, action_G, action_phase, action_tau, action_W):
        rate_list = []
        h_k_list = []
        rate_min = np.log2(1 + 10 ** (self.db_min / 10))
        sum_rate = 0
        sum_rate_for_jammer = 0
        sum_power_t = 0
        reward = 0
        reward_1 = 0
        P_in = 0
        U_sinr = np.zeros(self.K)
        u_rate_for_jammer = np.zeros(self.K)
        Theta = self.calculate_phase_shift_matrix(action_phase)
        H_BR = self.calculate_channel_matrix()
        H_JR = self.calculate_Jammer_RIS()
        H_RU = self.calculate_channel_matrix_RU()
        for u in range(self.K):
            U_sinr[u], sum_power_t, u_rate, signal_power = self.calculate_sinr(action_G, action_W, u, action_tau, Theta, H_BR, H_RU, H_JR)
            rate_list.append(u_rate)
            h_k_list.append(signal_power)
            if u_rate < rate_min:
                reward_1 = reward_1 + u_rate - rate_min
            sum_rate = sum_rate + u_rate

        rate_array = np.array(rate_list)
        if np.sum(rate_array) > 0:
            sum_rates = np.sum(rate_array)
            sum_squared_rates = np.sum(rate_array ** 2)
            jains_fairness = (sum_rates ** 2) / (self.K * sum_squared_rates)
        else:
            jains_fairness = 0.0

        reward = reward + min(rate_list)
        sum_power_in = self.yita * sum_power_t
        reward_2 = sum_power_in - self.pin_min
        if sum_power_in < self.pin_min:
            reward = reward + reward_1 / (reward_1 + reward_2) + reward_2 / (reward_1 + reward_2)
        next_state = self.get_state(sum_power_in, h_k_list)
        next_state_for_jammer = self.get_state_for_jammer(max(rate_list))
        return reward, next_state, 1/max(rate_list), next_state_for_jammer, sum_rate, jains_fairness, rate_array, sum_power_in


    def get_state(self, sum_power_in, h_k_list):
        state_list = []

        state_sum_power_in = np.array(sum_power_in, ndmin=1)
        state_h_k_list = np.array(h_k_list).flatten()
        state_P_in = np.array(self.pin_min, ndmin=1)

        state_list.append(state_sum_power_in)
        state_list.append(state_h_k_list)
        state_list.append(state_P_in)
        state_array = np.concatenate(state_list)

        return state_array

    def get_state_for_jammer(self, sum_rate_for_jammer):
        state_list_jammer = []

        state_sum_rate_for_jammer = np.array(sum_rate_for_jammer, ndmin=1)

        state_list_jammer.append(state_sum_rate_for_jammer)

        state_array_jammer = np.concatenate(state_list_jammer)

        return state_array_jammer