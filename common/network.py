import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Initialize  weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        input_dim = args.obs_dim + args.action_dim
        hidden_dim = 256
        # Q1 architecture
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear4 = nn.Linear(input_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        # initialize
        self.apply(weights_init)
    
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)      # sac.py에서 이미 tensor를 device로 보냄 -> CPU overhead 제거
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        # this is actually two critic networks
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, args, action_space=None):
        super(GaussianPolicy, self).__init__()
        input_dim = args.obs_dim
        num_actions = args.action_dim
        hidden_dim = 256
        self.min_log_std = -10.0
        self.max_log_std = 2.0
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init)
        
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high-action_space.low)/2.0)
            self.action_bias = torch.FloatTensor((action_space.high+action_space.low)/2.0)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Bound log-std to keep variance in a trainable range. -> nan_to_num 제거로 인한 GPU kernel 낭비 최소화
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        return mean, log_std
    
    def get_action(self, state):
        mean, log_std = self.forward(state)     # Tensor(1,1)
        std = log_std.exp()     # value section: exp([-10, 2]) --> [4e-5, 7.3891]
        normal = Normal(mean, std)
        # reparameterization trick, including noise
        x_t = normal.rsample()  # Gaussian dist에서 resampling，output：mean+std*sample
        # Enforcing Action Bound, a = tanh(u)
        y_t = torch.tanh(x_t)   # value section: [-1, 1]
        action = y_t*self.action_scale+self.action_bias     # value section: [-1, 1]
        # log miu(u|s)
        log_us = normal.log_prob(x_t)       # x_t log prob
        # log pi(a|s), a = tanh(u) enforcing action bound, change of variable from u to a
        log_prob = log_us - torch.log(1.000001-y_t.pow(2)) # y_t가 1이 되서 계산 안되는거 방지
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)*self.action_scale+self.action_bias  # value section: [-1, 1]
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
