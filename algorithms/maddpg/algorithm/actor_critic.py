import torch
import torch.nn as nn
import numpy as np
from global_matd3.utils.util import init, to_torch
from global_matd3.algorithms.utils.mlp import MLPBase
from global_matd3.algorithms.utils.act import ACTLayer


class MADDPG_Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, device):
        super(MADDPG_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # map observation input into input for rnn
        # self.mlp = MLPBase(args, obs_dim)
        # # self.fc1 = nn.Linear(obs_dim, 256)
        # # get action from rnn hidden state
        # self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, act_dim)
        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        self.output_activation = nn.functional.softmax
        self.to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, (int, float)):
            x = torch.tensor(x, dtype=torch.float32)  # 假设你需要 float32 类型的张量
        if isinstance(x, np.ndarray):
            x = torch.tensor(x.astype(np.float32), dtype=torch.float32)  # 确保 NumPy 数组是 float32 类型，然后转换
        elif isinstance(x, (int, float, list)):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(**self.tpdv)
        x = to_torch(x).to(**self.tpdv)
        # x = self.mlp(x)
        # # x = self.fc1(x)
        # # pass outputs through linear layer
        # action = self.act(x)
        # print(action)

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        action = self.output_activation(self.fc3(x), dim=-1)


        return action


class MADDPG_Critic(nn.Module):
    """
    Critic network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param central_obs_dim: (int) dimension of the centralized observation vector.
    :param central_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    """

    def __init__(self, args, central_obs_dim, central_act_dim, device, num_q_outs=1):
        super(MADDPG_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = central_obs_dim + central_act_dim

        self.mlp = MLPBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_outs = [init_(nn.Linear(self.hidden_size, 1)) for _ in range(num_q_outs)]
        for i, q_out in enumerate(self.q_outs):
            self.q_outs[i] = q_out.to(device)

        self.to(device)

    def forward(self, central_obs, central_act):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray) Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray) Centralized actions with which to compute Q-values.

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        central_obs = to_torch(central_obs).to(**self.tpdv)
        central_act = to_torch(central_act).to(**self.tpdv)

        x = torch.cat([central_obs, central_act], dim=1)

        x = self.mlp(x)
        q_values = [q_out(x) for q_out in self.q_outs]

        return q_values
