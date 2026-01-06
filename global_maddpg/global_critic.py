import torch as T
import torch.nn.functional as F
from global_maddpg.networks1 import CriticNetwork

class Global_Critic():
    def __init__(self, beta, input_dims, tau, n_actions, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_agents, update_actor_interval, noise, global_n_input, global_n_output):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.update_actor_iter = update_actor_interval
        self.learn_step_counter = 0
        self.noise = noise
        self.Global_Loss = []

        # Initialize global critics
        self.global_critic1 = CriticNetwork(beta, global_n_input, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                           n_actions=global_n_output, name='global_critic1', agent_label='global_critic1')
        self.global_critic2 = CriticNetwork(beta, global_n_input, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                           n_actions=global_n_output, name='global_critic2', agent_label='global_critic2')
        self.global_target_critic1 = CriticNetwork(beta, global_n_input, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                                  n_actions=global_n_output, name='global_target_critic1',
                                                  agent_label='global_target_critic1')
        self.global_target_critic2 = CriticNetwork(beta, global_n_input, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                                  n_actions=global_n_output, name='global_target_critic2',
                                                  agent_label='global_target_critic2')
        self.update_global_network_parameters(tau=1)

    def save_models(self):
        self.global_critic1.save_checkpoint()
        self.global_critic2.save_checkpoint()
        self.global_target_critic1.save_checkpoint()
        self.global_target_critic2.save_checkpoint()

    def load_models(self):
        self.global_critic1.load_checkpoint()
        self.global_critic2.load_checkpoint()
        self.global_target_critic1.load_checkpoint()
        self.global_target_critic2.load_checkpoint()

    def global_learn(self, agents_nets, state, action, reward_g, reward_l, state_, terminal):
        self.agents_networks = agents_nets

        # Convert inputs to tensors
        states = T.tensor(state, dtype=T.float).to(self.global_critic1.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.global_critic1.device)
        actions = T.tensor(action, dtype=T.float).to(self.global_critic1.device)
        rewards_g = T.tensor(reward_g, dtype=T.float).to(self.global_critic1.device)
        rewards_l = T.tensor(reward_l, dtype=T.float).to(self.global_critic1.device)
        done = T.tensor(terminal).to(self.global_critic1.device)

        # Evaluate target actions using both global and local target actors
        target_actions = []
        for i in range(self.number_agents):
            local_target_action = agents_nets[i].target_actor.forward(
                states_[:, i * self.number_states:(i + 1) * self.number_states])
            global_target_action = self.global_target_critic1.forward(
                states_, actions).detach()  # Use global critic to guide target actions
            combined_target_action = (local_target_action + global_target_action) / 2  # Combine global and local
            target_actions.append(combined_target_action)
        target_actions = T.cat(target_actions, dim=1)

        # Add noise and clamp actions
        target_actions = target_actions + T.clamp((T.randn_like(target_actions) * self.noise), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -0.999, 0.999)

        # Compute target Q-values
        q1_ = self.global_target_critic1.forward(states_, target_actions)
        q2_ = self.global_target_critic2.forward(states_, target_actions)
        q1_[done] = 0.0
        q2_[done] = 0.0
        critic_value_ = T.min(q1_, q2_)
        target = rewards_g + self.gamma * critic_value_

        # Compute critic loss
        q1 = self.global_critic1.forward(states, actions)
        q2 = self.global_critic2.forward(states, actions)
        critic_loss = F.mse_loss(target, q1) + F.mse_loss(target, q2)
        self.Global_Loss.append(critic_loss.detach().cpu().numpy())

        # Update global critics
        self.global_critic1.optimizer.zero_grad()
        self.global_critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.global_critic1.optimizer.step()
        self.global_critic2.optimizer.step()

        # Update target networks
        self.update_global_network_parameters()

        # Update local actors using actor_global_loss
        if self.learn_step_counter % self.update_actor_iter == 0:
            actor_global_loss = -self.global_critic1.forward(states, actions).mean()
            for i in range(self.number_agents):
                if i == self.number_agents - 1:
                    agents_nets[i].local_learn(actor_global_loss.detach(),  # Detach to avoid gradient propagation
                                                states[:, i * self.number_states:(i + 1) * self.number_states],
                                                actions[:, (i - 1) * self.number_actions[0]:(
                                                                    (i - 1) * self.number_actions[0] +
                                                                    self.number_actions[1])],
                                                rewards_l[:, i], states_[:, i * self.number_states:(i + 1) * self.number_states],
                                                done)
                else:
                    agents_nets[i].local_learn(actor_global_loss.detach(),  # Detach to avoid gradient propagation
                                                states[:, i * self.number_states:(i + 1) * self.number_states],
                                                actions[:, i * self.number_actions[0]: (i + 1) * self.number_actions[0]],
                                                rewards_l[:, i],
                                                states_[:, i * self.number_states:(i + 1) * self.number_states],
                                                done)

        self.learn_step_counter += 1

    def update_global_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.global_target_critic1.parameters(), self.global_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.global_target_critic2.parameters(), self.global_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)