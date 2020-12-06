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
from sac.infrastructure.utils import Path

from airl import AIRL
from dataset import DemoDataset

class AIRL_Trainer(object):
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
        agent_params = {}
        agent_params['alpha'] = self.params['alpha']
        agent_params['gamma'] = self.params['gamma']
        agent_params['polyak'] = self.params['polyak']
        agent_params['batch_size'] = self.params['p_batch_size']
        agent_params['learning_rate'] = self.params['p_learning_rate']
        agent_params['rb_size'] = self.params['rb_size']
        agent_params['n_layers'] = self.params['p_n_layers']
        agent_params['size'] = self.params['p_size']
        agent_params['ep_len'] = self.params['ep_len']
        agent_params['start_steps'] = self.params['start_steps']
        self.policy = SACAgent(self.env, agent_params)

        #############
        ## AIRL
        #############

        self.expert_traj = DemoDataset().get_trajs()
        self.irl_model = AIRL(self.env, self.expert_traj, self.params)



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

        returns = []
        for itr in range(self.params['iterations']):
            itr_start_time = time.time()
            print('\n----------------itr #%d ----------------- ' % itr)
            print('\nObtaining samples (IRL).....')

            paths = utils.sample_n_trajectories(self.test_env, self.policy, self.params['irl_batch_size'], self.params['ep_len']) #execute policy to collect trajectories

            print("\nRunning IRL...")
            last_irl_loss, irl_probs = self.compute_irl(paths) #train discriminator

            print("\nOptimizing policy...")

            all_p_logs = self.sac_train_subloop() # update policy

            print('\nTime: %f' % (time.time() - start_time))
            print('\nItrTime %f' % (time.time() - itr_start_time))

            print('\nLogging...')
            self.perform_logging(epoch, self.policy, all_p_logs, last_irl_loss, irl_probs)
            if itr > 0 and self.params['save_params'] and itr % self.params['save_every'] == 0:
                print('\nSaving...')
                self.irl_model.save(self.params['logdir'], itr)
                self.agent.save(self.params['logdir'], itr)

        return 

    ####################################
    ####################################

    def sac_train_subloop(self):
        for t in range(self.params['steps_per_epoch']):
            self.total_envsteps += 1

            self.step_with_irl(True, self.irl_model)

            if t >= self.params['learning_starts'] and t % self.params['update_every'] == 0:
                all_logs = self.train_agent()

    def train_policy(self):
        all_logs = []
        for train_step in range(self.params['update_every']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['p_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    def compute_irl(self, paths):
        last_loss = self.irl_model.fit(paths, self.policy)
        probs = self.irl_model.eval(paths, gamma=self.discount, itr=itr)
        return last_loss, probs

    ####################################
    ####################################

    def perform_logging(self, itr, eval_policy, policy_logs, irl_loss, irl_probs):

        last_log = policy_logs[-1]

        #######################

        # # collect eval trajectories, for logging
        # print("\nCollecting data for eval...")
        # eval_paths = utils.sample_n_trajectories(self.test_env, eval_policy, self.params['eval_num_episodes'], self.params['ep_len'])

        # #######################

        # # save eval metrics
        # eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

        # eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        # # decide what to log
        # logs = OrderedDict()
        # logs["Eval_AverageReturn"] = np.mean(eval_returns)
        # logs["Eval_StdReturn"] = np.std(eval_returns)
        # logs["Eval_MaxReturn"] = np.max(eval_returns)
        # logs["Eval_MinReturn"] = np.min(eval_returns)

        logs['Iteration'] = itr
        logs["Train_EnvstepsSoFar"] = self.total_envsteps
        logs["TimeSinceStart"] = time.time() - self.start_time
        logs["IRL Loss"] = irl_loss
        logs['IRLRewardMean']= np.mean(probs)
        logs['IRLRewardMax'] = np.max(probs)
        logs['IRLRewardMin'] = np.min(probs)
        logs.update(last_log)

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')

        self.logger.flush()

