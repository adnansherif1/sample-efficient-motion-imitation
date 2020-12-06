import torch
import numpy as np
from discrim import *
from sac.infrastructure.pytorch_util import from_numpy, to_numpy
from fusion import *

class AIRL(object):
    """ 
    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env, expert_trajs, params):

        super(AIRL, self).__init__()

        self.obs_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.score_discrim = params['score_discriminator']
        self.gamma = params['gamma']

        self.demos = expert_trajs

        self.max_itrs = self.params['irl_steps_per_itr']

        self.lr = params['irl_learning_rate']
        self.discrim = Discriminator(self.obs_dim, self.obs_dim, self.lr, self.gamma)

        self.fusion = RamFusionDistr(100, subsample_ratio=0.5)

        self.expert_trajs = expert_trajs

    def fit(self, paths, policy,**kwargs):
        # get samples under current policy with log probs
        policy.eval_prob(paths)

        # training on mixed data makes reward more robust at end of training
        old_paths = self.fusion.sample_paths(n=len(paths)) 
        self.fusion.add_paths(paths)
        paths = paths+old_paths

        # eval expert log probs under current policy
        expert_probs = policy.eval_prob(self.expert_trajs)

        obs, obs_next, acts, path_probs = \
            self.extract_paths(paths,
                               keys=('observation', 'next_observation', 'action', 'a_logprob'))
        expert_obs, expert_obs_next, expert_acts, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('observation', 'next_observation', 'action', 'a_logprob'))

        # Train discriminator
        for it in range(self.max_itrs):
            nobs_batch, obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts, expert_probs, batch_size=batch_size)

            # Build feed dict
            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)

            loss = self.discrim.update(obs_batch, act_batch, nobs_batch, lprobs_batch, labels)

        return loss.item()

    @staticmethod
    def sample_batch(*args, batch_size=32):
        N = args[0].shape[0]
        batch_idxs = np.random.randint(0, N, batch_size)  # trajectories are negatives
        return [data[batch_idxs] for data in args]

    @staticmethod
    def extract_paths(paths, keys=('observation', 'action'), stack=True):
        if stack:
            return [np.stack([t[key] for t in paths]).astype(np.float32) for key in keys]
        else:
            return [np.concatenate([t[key] for t in paths]).astype(np.float32) for key in keys]

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=('observation', 'next_observation', 'action', 'a_logprob')) # log prob should already be calculated by fit
            path_probs = np.expand_dims(path_probs, axis=1)

            obs = from_numpy(obs)
            obs_next = from_numpy(obs_next)
            acts = from_numpy(acts)
            path_probs = from_numpy(path_probs)

            scores = self.discrim(obs, acts, obs_next, path_probs)[0]
            score = np.log(scores) - np.log(1-scores)
            score = score[:,0]
        else:
            obs, acts = self.extract_paths(paths)
            obs = from_numpy(obs)
            acts = from_numpy(acts)
            reward = self.discrim.get_reward(obs, acts)
            score = reward[:,0]
        return to_numpy(score)

    def eval_single(self, obs):
        obs = from_numpy(obs)
        reward = self.discrim.get_reward(obs, None)
        print(reward.shape)
        # score = reward[:, 0]
        return to_numpy(score)

    # def debug_eval(self, paths, **kwargs):
    #     obs, acts = self.extract_paths(paths)
    #     reward, v, qfn = tf.get_default_session().run([self.reward, self.value_fn,
    #                                                         self.qfn],
    #                                                   feed_dict={self.act_t: acts, self.obs_t: obs})
    #     return {
    #         'reward': reward,
    #         'value': v,
    #         'qfn': qfn,
    #     }