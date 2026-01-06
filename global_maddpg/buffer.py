import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float16)
        self.action_memory = np.zeros((self.mem_size, n_actions[0] * (n_agents - 1) + n_actions[1]), dtype=np.float16)
        self.reward_global_memory = np.zeros(self.mem_size)
        self.reward_local_memory =  np.zeros((self.mem_size, n_agents), dtype=np.float16)
        self.new_state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float16)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward_g, reward_l, state_, done):
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a NumPy array")

            # 检查 action 的形状是否正确
        expected_shape = (self.action_memory.shape[1],)  # 获取 action_memory 的第二维大小
        if action.shape != expected_shape:
            raise ValueError(f"Action shape must be {expected_shape}, but got {action.shape}")

        # 检查 action 的数据类型是否正确（可选，因为 NumPy 在赋值时会自动进行类型转换，但最好明确）
        if action.dtype != self.action_memory.dtype:
            # 注意：这里我们没有直接抛出错误，因为 NumPy 通常会在赋值时处理类型转换。
            # 但是，如果转换不是无损的（例如，从 float64 到 float16），则可能会丢失精度。
            # 如果您希望确保精度不丢失，可以添加错误处理逻辑。
            action = action.astype(self.action_memory.dtype)  # 显式转换数据类型
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_global_memory[index] = reward_g
        self.reward_local_memory[index] = reward_l
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards_g = self.reward_global_memory[batch]
        rewards_l = self.reward_local_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards_g, rewards_l, states_, dones