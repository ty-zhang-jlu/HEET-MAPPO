import numpy as np
from global_mappo_8.envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        self.movable = True

        self.action_space = [[] for _ in range(self.num_agent)]
        self.observation_space = [[] for _ in range(self.num_agent)]
        self.share_observation_space = np.zeros((self.num_agent, self.signal_obs_dim), dtype=np.float32)  # 假设所有智能体共享一个全局观察空间

        # 初始化动作空间和观察空间
        for i in range(self.num_agent):
            # 动作空间：这里我们假设动作是浮点数，范围在-inf到+inf之间
            # 在实际应用中，你可能需要根据具体任务定义更具体的动作范围或类型
            action_range = (-1, 1)
            self.action_space[i] = action_range  # 这里简化为一个范围，而不是一个完整的空间对象

            # 观察空间：同样地，我们假设观察是浮点数数组，范围在-inf到+inf之间
            # 在实际应用中，你可能需要定义更具体的观察形状或类型
            observation_shape = (self.signal_obs_dim,)
            self.observation_space[i] = observation_shape  # 这里简化为一个形状，而不是一个完整的空间对象



    def step(self, action_G, action_phase, action_tau, action_W):
        # 输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        results = self.env.step(action_G, action_phase,action_tau, action_W)
        obs, rews, dones, infos, j_ob, j_rew = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, j_ob, j_rew

    def reset(self):
        obs, j_ob = self.env.reset()
        return np.stack(obs), j_ob

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
