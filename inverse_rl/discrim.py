import torch
import torch.nn as nn
from sac.infrastructure.pytorch_util import from_numpy
import os

class RewardApproximator(nn.Module):
    def __init__(self, dim, hidden_dims=[32,32]):
        super(RewardApproximator, self).__init__()

        layers = [nn.Linear(dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ValueFn(nn.Module):
    def __init__(self, dim, hidden_dims=[32,32]):
        super(ValueFn, self).__init__()

        layers = [nn.Linear(dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)

class Discriminator(nn.Module):
    def __init__(self, rew_dim, v_dim, lr=1e-3, state_only=True, discount=0.99):
        super(Discriminator, self).__init__()
        self.reward_arch = RewardApproximator(rew_dim)
        self.value_fn_arch = ValueFn(v_dim)

        self.r_optimizer = torch.optim.Adam(
            self.reward_arch.parameters(),
            lr,
        )
        self.v_optimizer = torch.optim.Adam(
            self.value_fn_arch.parameters(),
            lr,
        )

        self.state_only = state_only
        self.gamma = discount

    def get_reward(self, obs_t, act_t):
        if not self.state_only:
            x = torch.cat([obs_t, act_t], dim=1)
        else:
            x = obs_t
        return self.reward_arch(x)

    def forward(self, obs_t, act_t, nobs_t, log_q_tau):
        if not self.state_only:
            x = torch.cat([obs_t, act_t], dim=1)
        else:
            x = obs_t
        r_t = self.reward_arch(x)
        v_t = self.value_fn_arch(obs_t)
        v_tp1 = self.value_fn_arch(nobs_t)
        q_t = r_t + self.gamma*v_tp1 
        log_p_tau = q_t - v_t   # = f(s,a) = log( exp ( f(s,a) ))

        log_pq = torch.logsumexp(torch.cat([log_p_tau, log_q_tau]), dim=1) # = element wise exp(f) + exp(log(pi)) = element wise exp(f) + pi

        discrim_output = torch.exp(log_p_tau-log_pq)  # = log ( exp (f)) - log(exp(f)+ pi) = log(exp(f)/(exp(f) + pi))

        return discrim_output, log_p_tau, log_pq

    def update(self, obs_t, act_t, nobs_t, log_q_tau, labels):
        obs_t = from_numpy(obs_t)
        act_t = from_numpy(act_t)
        nobs_t = from_numpy(nobs_t)
        log_q_tau = from_numpy(log_q_tau)
        labels = from_numpy(labels)

        _, log_p_tau, log_pq = self(obs_t, act_t, nobs_t, log_p_tau, labels)

        loss = -torch.mean(labels*(log_p_tau-log_pq) + (1-labels)*(log_q_tau-log_pq))

        self.r_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        loss.backward()
        self.r_optimizer.step()
        self.v_optimizer.step()

        return loss.item()

    def save(self, path, itr):
        save_str = path + "/" + str(self) + "_" + str(itr) + ".pth"
        torch.save(self.state_dict(), save_str)

    def load(self, path, itr):
        save_str = path + "/" + str(self) + "_" + str(itr) + ".pth"
        assert(os.path.exists(save_str))
        return torch.load(save_str)

    def __str__(self):
        return "discrim"