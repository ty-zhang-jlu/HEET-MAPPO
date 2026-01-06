"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]  # 每个envs的参数是一样的
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.action_G = None
        self.action_phase = None
        self.action_tau = None
        self.action_W = None
        self.M = 4
        self.K = 5
        self.action_dim = [2, 2 * self.M * self.K]

    def step(self, action_G, action_phase, action_tau, action_W):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(action_G, action_phase, action_tau, action_W)
        return self.step_wait()

    def step_async(self, action_G, action_phase, action_tau, action_W):
        self.action_G = None
        self.action_phase = None
        self.action_tau = None
        self.action_W = None

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.action_G, self.action_phase, self.action_tau, self.action_W, self.envs)]
        obs, rews, dones, infos, j_ob, j_rew = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.action_G = None
        self.action_phase = None
        self.action_tau = None
        self.action_W = None
        return obs, rews, dones, infos, j_ob, j_rew

    def reset(self):
        obs, j_ob = [env.reset() for env in self.envs] # [env_num, agent_num, obs_dim]
        return np.array(obs), j_ob

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError