import math
import numpy as np
import time
from global_maddpg.Environment import Environment

from global_maddpg.ddpg_torch import Agent
from global_maddpg.buffer import ReplayBuffer
from global_maddpg.global_critic import Global_Critic
from global_maddpg.OPT_DDPG import Agent_J
from global_maddpg.utils import to_tensor_var
import matplotlib.pyplot as plt
import os
import csv

directory_path = '/tmp/pycharm_project_518/global_maddpg/model2/maddpg'
os.makedirs(directory_path, exist_ok=True)



MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 1
NUMBER_OF_EVAL_EPISODES = 50
n_episodes = 0
step_per_episode = 20
M = 4
J = 3
N = 16
K = 5
n_input_j = 1
n_output_j = 2 * J * K
alpha = 0.0001
beta = 0.001
tau = 0.005
n_agents = N + 1
use_cuda = False
env = Environment(M, N, K, Z=5, c1=11.95, c2=0.136, kappa=0.15, dR=0.1, dB=0.2, lambda_c=35/3, P_tx=400,
                      sigma_noise=10 ** (-13), RIS_height=500, Sigma2BR=0.01, Sigma2RU=0.01, K_BR=3.5, K_RU=2.2,
                      Sigma_csi=0.02, beta_min=0.2, theta_bar=0.1, kappa_bar=0.4, db_min=2, pin_min=40, yita=0.7,
                   Jammer_position=[1000, 1000, 200], P_Jammer=0.5, J=3, Sigma_csi_Jammer=0.02, dJ=0.2)
agent_j = Agent_J(alpha, beta, n_input_j, tau, n_output_j, gamma=0.99, max_size=10000, C_fc1_dims=512,
                   C_fc2_dims=256, C_fc3_dims=128, A_fc1_dims=256, A_fc2_dims=128, batch_size=64, n_agents=1)


##---Initializations networks parameters---##
batch_size = 32
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.005
update_actor_interval = 2
noise = 0.2
# actor and critic hidden layers
C_fc1_dims = 256
C_fc2_dims = 128
C_fc3_dims = 64
A_fc1_dims = 256
A_fc2_dims = 128
marl_n_input = 2 + K  # 每个agent的状态
marl_n_output = [2, 40]
global_n_input = marl_n_input * n_agents
global_n_output = marl_n_output[0] * (n_agents - 1) + marl_n_output[1]
#--------------------------------------------
agents = []
for index_agent in range(n_agents):
    print("Initializing agent", index_agent)
    if index_agent == n_agents - 1:
        agent = Agent(alpha, beta, marl_n_input, tau, marl_n_output[1], gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, batch_size, n_agents, index_agent, noise)
    else:
        agent = Agent(alpha, beta, marl_n_input, tau, marl_n_output[0], gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
              A_fc1_dims, A_fc2_dims, batch_size, n_agents, index_agent, noise)

    agents.append(agent)
memory = ReplayBuffer(memory_size, marl_n_input, marl_n_output, n_agents)
print("Initializing Global critic ...")
global_agent = Global_Critic(beta, marl_n_input, tau, marl_n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_agents, update_actor_interval, noise, global_n_input, global_n_output)

##Let's go
Training_episode_rewards = []
Training_episode_rates = []
record_critics_loss_ = np.zeros([n_agents + 1, MAX_EPISODES])
env_state, env_state_j = env.initial_state()
start = time.perf_counter()
while n_episodes < MAX_EPISODES - 1:
    T = 0
    done = False
    while not done:
        state_old, state_old_j = env_state, env_state_j
        action_phase = []
        action_tau = []
        rewards = []
        rates = []
        actor_action = []

        start = time.perf_counter()
        for i in range(n_agents):
            action = agents[i].choose_action(env_state)

            if i == n_agents - 1:
                action_BS = np.clip(action, -0.999, 0.999).reshape(-1)
                g_theta = action_BS[0: M * K] * math.pi
                actor_action.extend(g_theta.tolist())
                g_beta = (action_BS[M * K: 2 * M * K] + 1) / 2
                actor_action.extend(g_beta.tolist())
                g_array = np.cos(g_theta) * g_beta + np.sin(g_theta) * g_beta * 1j
                action_G = np.reshape(g_array, (M, K))
                action_array = np.array(actor_action)
            else:
                action_phase.append(math.floor(((action[0] + 1) / 2) * (2 ** 3 + 1)) * (2 * math.pi / (2 ** 3)))
                actor_action.append(math.floor(((action[0] + 1) / 2) * (2 ** 3 + 1)) * (2 * math.pi / (2 ** 3)))
                action_tau.append((action[1] + 1) / 2)
                actor_action.append((action[1] + 1) / 2)
        end = time.perf_counter()
        runTime = end - start
        # print(runTime)
        action_j = agent_j.choose_action(np.asarray(state_old_j).flatten())
        w_theta = action_j[0: J * K] * math.pi
        w_beta = (action_j[J * K: 2 * J * K] + 1) / 2
        w_array = np.cos(w_theta) * w_beta + np.sin(w_theta) * w_beta * 1j
        action_W = np.reshape(w_array, (J, K))

        reward, state_new_all, reward_j, state_new_all_j, sum_rate = env.calculate_reward(action_G, action_phase, np.array(action_tau), action_W)
        # print('每个T上的reward:', reward)
        rewards.append(reward)  # 依次存储在每次迭代的每个T上产生的所有的reward
        rates.append(sum_rate)
        T = T + 1
        if T == step_per_episode - 1:
            done = True
        if done:
            Training_episode_rewards.append(np.mean(np.array(rewards)))  # 这个变量用来存储每次episode的奖励
            print('第', n_episodes, '次训练的平均奖励是：', np.mean(np.array(rewards)))  # 输出本次episode上的奖励就是平均奖励
            Training_episode_rates.append(np.mean(np.array(rates)))  # 这个变量用来存储每次episode的奖励
            # print('第', n_episodes, '次训练的平均奖励是：', np.mean(np.array(rates)))
            n_episodes += 1
        else:
            env_state = state_new_all
            env_state_j = state_new_all_j

        state_old = to_tensor_var(state_old, use_cuda).unsqueeze(0).repeat(n_agents, 1)
        rewards = to_tensor_var(reward, use_cuda).unsqueeze(0).repeat(n_agents, 1)
        state_new_all = to_tensor_var(state_new_all, use_cuda).unsqueeze(0).repeat(n_agents, 1)

        # taking the agents actions, states and reward
        memory.store_transition(np.asarray(state_old).flatten(), np.asarray(action_array).flatten(),
                                reward, np.asarray(rewards).flatten(), np.asarray(state_new_all).flatten(), done)
        agent_j.remember(np.asarray(state_old_j).flatten(), np.asarray(action_j).flatten(),
                              reward_j, np.asarray(state_new_all_j).flatten(), done)

        if n_episodes >= EPISODES_BEFORE_TRAIN:
            agent_j.learn()
        if memory.mem_cntr >= batch_size:
            states, actions, rewards_g, rewards_l, states_, dones = memory.sample_buffer(batch_size)
            global_agent.global_learn(agents, states, actions, rewards_g, rewards_l, states_, dones)


    record_critics_loss_[0, n_episodes] = np.mean(np.asarray(global_agent.Global_Loss))
    global_agent.Global_Loss = []

    for i in range(n_agents):
        record_critics_loss_[i + 1, n_episodes] = np.mean(np.asarray(agents[i].local_critic_loss))
        agents[i].local_critic_loss = []

    # if n_episodes % 500 == 0 and n_episodes != 0:
    #     global_agent.save_models()
    #     for i in range(n_agents):
    #      agents[i].save_models()

end = time.perf_counter()
runTime = end - start
print("Running Time：", runTime, "s")

filename = 'MADDPG_Reward_10.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    for item in Training_episode_rewards:
        writer.writerow([item])
# filename = 'MADDPG_Rate_1000_1.csv'
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for item in Training_episode_rates:
#         writer.writerow([item])
print('保存数据成功MADDPG_Reward_10')



