# adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py

from copy import deepcopy
from collections import OrderedDict

from sac.infrastructure.replay_buffer import ReplayBuffer
from sac.infrastructure.utils import *
from .base_agent import BaseAgent

import torch
from torch import nn
from torch import optim

from sac.infrastructure import pytorch_util as ptu
from .sac_policy import MLPActorCritic
import itertools
import os

class SACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.alpha = self.agent_params['alpha']
        self.polyak = self.agent_params['polyak']
        self.ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=[self.agent_params['size'] for i in range(self.agent_params['n_layers'])])
        self.ac_target = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        self.ac.to(ptu.device)
        self.ac_target.to(ptu.device)

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], self.agent_params['rb_size'])

        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.agent_params['learning_rate'])
        self.q_optimizer = optim.Adam(self.q_params, lr=self.agent_params['learning_rate'])

        self.t = 0
        self.last_obs = self.env.reset()
        self.ep_len = 0

    def compute_loss_q(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        q1 = self.ac.q1(ob_no, ac_na)
        q2 = self.ac.q2(ob_no, ac_na)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(next_ob_no)

            # Target Q-values
            q1_pi_targ = self.ac_target.q1(next_ob_no, a2)
            q2_pi_targ = self.ac_target.q2(next_ob_no, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = re_n + self.gamma * (1 - terminal_n) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=ptu.to_numpy(q1),
                      Q2Vals=ptu.to_numpy(q2))

        return loss_q, q_info

    def compute_loss_pi(self, ob_no):
        pi, logp_pi = self.ac.pi(ob_no)
        q1_pi = self.ac.q1(ob_no, pi)
        q2_pi = self.ac.q2(ob_no, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=ptu.to_numpy(logp_pi))

        return loss_pi, pi_info

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(ob_no, ac_na, re_n, next_ob_no, terminal_n)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(ob_no)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        loss = OrderedDict()
        loss['Q loss'] = loss_q.item()
        loss['Policy Loss'] = loss_pi.item()
        # loss['Q1 value'] = q_info['Q1Vals']
        # loss['Q2 value'] = q_info['Q2Vals']
        # loss['Log prob pi'] = pi_info['LogPi']
        return loss

    def sample(self, batch_size):
        return self.replay_buffer.sample_batch(batch_size)

    def step(self, use_irl=False, irl_model=None):
        if self.t < self.agent_params['start_steps']:
            action = self.env.action_space.sample()
        else:
            action = self.ac.get_action(self.last_obs)

        next_obs, reward, done, info = self.env.step(action)

        if use_irl:
            reward = irl_model.eval_single(self.last_obs)

        self.ep_len += 1

        done = 0 if self.ep_len == self.agent_params['ep_len'] else done
        self.replay_buffer.store(self.last_obs, action, reward, next_obs, done)
        s = self.last_obs
        self.last_obs = next_obs.copy()

        if done or self.ep_len == self.agent_params['ep_len']:
            self.last_obs = self.env.reset()
            self.ep_len = 0
        self.t += 1

    def get_action(self, obs, deterministic=False):
        return self.ac.get_action(obs, deterministic)

    def eval_prob(self, paths, insert_key='a_logprob'):
        for path in paths:
            obs = ptu.from_numpy(path['observation'])
            ac = ptu.from_numpy(path['action'])
            path_probs = ptu.to_numpy(self.ac.pi.get_prob(obs, ac))
            path[insert_key] = path_probs

    def save(self, path, itr):
        save_str = path + "/" + str(self) + "_" + str(itr) + ".pth"
        torch.save({
            'ac_state_dict': self.ac.state_dict(),
            'ac_target_state_dict': self.ac_target.state_dict(),
            'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'replay_buffer_state': self.replay_buffer.state()
            }, save_str)

    def load(self, path, itr):
        save_str = path + "/" + str(self) + "_" + str(itr) + ".pth"
        assert(os.path.exists(save_str))
        checkpoint = torch.load(save_str)
        self.ac.load_state_dict(checkpoint['ac_state_dict'])
        self.ac_target.load_state_dict(checkpoint['ac_target_state_dict'])
        self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.replay_buffer.load_state(checkpoint['replay_buffer_state'])

    def __str__(self):
        return "sac_agent"