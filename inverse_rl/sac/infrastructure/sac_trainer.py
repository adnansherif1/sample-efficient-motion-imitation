from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
import sac.infrastructure.pytorch_util as ptu

import sac.infrastructure.utils as utils
from sac.infrastructure.logger import Logger
from sac.models.sac_agent import SACAgent
from sac.infrastructure.sac_utils import register_custom_envs

class SAC_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        register_custom_envs()
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)
        self.test_env = gym.make(self.params['env_name'])

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

        #############
        ## AGENT
        #############

        self.agent = SACAgent(self.env, self.params)

    def run_training_loop(self):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        self.start_time = time.time()

        total_steps = self.params['epochs'] * self.params['steps_per_epoch']

        for t in range(total_steps):
            self.total_envsteps = t

            epoch = (t//self.params['steps_per_epoch'])
            if t % self.params['steps_per_epoch'] == 0:
                print("\n\n********** Iteration %i ************"%epoch)

            self.agent.step()

            if t >= self.params['learning_starts'] and t % self.params['update_every'] == 0:
                all_logs = self.train_agent()

            if t % self.params['steps_per_epoch'] == 0 and epoch > 0:
                print('\nLogging...')
                self.perform_logging(epoch, self.agent.ac, all_logs)
                if self.params['save_params'] and epoch > 1 and (epoch-1) % self.params['save_every'] == 0:
                    print('\nSaving... ')
                    self.agent.save(self.params['logdir'], epoch)

    ####################################
    ####################################

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['update_every']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, eval_policy, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_n_trajectories(self.test_env, eval_policy, self.params['eval_num_episodes'], self.params['ep_len'])

        #######################

        # save eval metrics
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        # decide what to log
        logs = OrderedDict()
        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)

        logs['Epoch'] = self.total_envsteps // self.params['steps_per_epoch']
        logs["Train_EnvstepsSoFar"] = self.total_envsteps
        logs["TimeSinceStart"] = time.time() - self.start_time
        logs.update(last_log)

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')

        self.logger.flush()
