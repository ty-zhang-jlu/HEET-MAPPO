import numpy as np
import torch as T
import torch.nn.functional as F
from global_maddpg.networks1 import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims, A_fc1_dims,
                 A_fc2_dims, batch_size, n_agents, agent_name, noise):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.agent_name = agent_name
        self.noise = noise
        self.local_critic_loss = []

        # Initialize networks
        self.actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='actor', agent_label=agent_name)
        self.critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                n_actions=n_actions, name='critic', agent_label=agent_name)
        self.target_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='target_actor', agent_label=agent_name)
        self.target_critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                n_actions=n_actions, name='target_critic', agent_label=agent_name)
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=self.number_actions),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def local_learn(self, global_loss, state, action, reward_l, state_, terminal):
        states = state
        states_ = state_
        actions = action
        rewards = reward_l
        done = terminal

        # Compute target actions
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)
        critic_value_[done] = 0.0
        target = rewards + self.gamma * critic_value_

        # Update critic
        critic_loss = F.mse_loss(target, critic_value)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        self.local_critic_loss.append(critic_loss.detach().cpu().numpy())

        # Update actor
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        actor_loss += 0.1 * global_loss  # Add global loss as a regularization term
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)