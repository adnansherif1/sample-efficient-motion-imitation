import json
import glob
import numpy as np
from gym import spaces
import gym
import os
from sac.infrastructure.utils import Path

VELOCITY_LIMITING = 1/10
AUDIO_FEATURE_HIGH = 500

class DemoDataset():
	def __init__(self, path='data/DanceRevolution/trajectories.npy'):
		print("Loading expert trajectories...")
		dataset = np.load(path, allow_pickle=True)
		self.obs_dim = dataset['obs_dim']
		self.audio_dim = dataset['audio_dim']
		self.pose_min = dataset['pose_min']
		self.pose_max = dataset['pose_max']
		self.pose_accuracies = dataset['pose_accuracies']
		self.num_kps = dataset['num_kps']

		self.ac_dim = self.obs_dim - self.audio_dim

		self.trajs = []
		for traj in dataset['trajs']:  # format of traj is specified in data_to_traj, calculations below are just indexing (s,a,s') out of the concatenated dataset
			obs = traj[:, :self.obs_dim]
			ac = traj[:, self.obs_dim:self.obs_dim+self.ac_dim]
			n_obs = traj[:, self.obs_dim+self.ac_dim:]

			terminal = np.zeros(obs.shape[0])
			terminal[-1] = 1
			
			self.trajs.append(Path(obs, [], ac, [], n_obs, terminal))
		print("Loaded")

	def get_new_traj(self, seed=None):
		traj_idx = np.random.randint(len(self.traj), seed=seed)
		return self.traj[traj_idx]

	def get_trajs(self):
		return self.trajs

# TODO: init currently takes like 5-10 mins; might want to parallelize
class HipHop(gym.Env):
	def __init__(self, path='/home/jszou/cs/cs285/sac/envs/dance_rev/trajectories.npz', clip_length=30*30, video_res=(640,320), weight_reward=True):
		super(HipHop, self).__init__()		

		self.dataset = DemoDataset(path) # even though env and expert trajs usually separate, we share the data to miminimize memory usage

		# setup state variables
		self.t = 0
		self.track = None
		self.track_t = 0
		self.track_idx = -1
		self.clip_length = clip_length
		self.pose = np.empty(self.dataset.ac_dim)
		self.weight_reward = weight_reward

		# interface with gym 
		self.action_space = spaces.Box(low=self.dataset.pose_min*VELOCITY_LIMITING, high=self.dataset.pose_max*VELOCITY_LIMITING, shape=(self.dataset.ac_dim,))
		self.observation_space = spaces.Box(low=np.ones(self.dataset.obs_dim)*float("-inf"), high=np.ones(self.dataset.obs_dim)*float("inf"), dtype=np.float32) # nonsensical limits because we only create this to fit openai gym format
		
	def step(self, pose_delta):
		assert(self.track is not None and self.t < self.track_t and self.t < self.clip_length), "env needs to be reset"
		assert(pose_delta.shape[0] == self.dataset.ac_dim), "input dim different than expected"

		self.pose += pose_delta

		# keep pose within "video" bounds
		self.pose = np.maximum(self.pose, self.dataset.pose_min)
		self.pose = np.minimum(self.pose, self.dataset.pose_max)

		self.t += 1

		done = False
		audio = np.zeros(self.dataset.audio_dim)
		if self.t >= self.clip_length or self.t >= self.track_t: #TODO: add some early termination thing like joint touching ground, etc.
			done = True
		else:
			audio = self.track[self.t, self.dataset.ac_dim:self.dataset.obs_dim]

		return np.append(self.pose, audio), self.get_reward(), done, {}

	def get_reward(self):
		pose = self.pose.reshape(self.dataset.num_kps, -1)
		ref_pose = self.track[self.t, :self.dataset.ac_dim].reshape(self.dataset.num_kps, -1)
		if self.weight_reward:
			weights = self.dataset.pose_accuracies[self.track_idx][self.t]
			return -np.mean(weights*np.sqrt(np.sum((pose-ref_pose)**2, axis=1)))
		else:
			return -np.mean(np.sqrt(np.sum((pose-ref_pose)**2, axis=1)))

	def seed(self, seed):
		np.random.seed(seed)

	# TODO: another function for reset to random human pose or reset to DeepMimic base pose

	def reset(self):
		self.t = 0
		trajs = self.dataset.get_trajs()
		self.track_idx = np.random.randint(len(trajs))
		expert_traj = trajs[self.track_idx]
		self.track = expert_traj['observation'].copy()
		self.pose = expert_traj['observation'][0, :self.dataset.ac_dim].copy()
		self.track_t = len(self.track)
		return np.concatenate([self.pose, self.track[self.t, self.dataset.ac_dim:self.dataset.obs_dim]], axis=0)

	def reset_ref(self, ref_pose, song_idx=None):
		self.pose = ref_pose
		trajs = self.dataset.get_trajs()
		if song_idk is not None:
			expert_traj = trajs[song_idx]
			self.track_idx = song_idx 
		else:
			self.track_idx = np.random.randint(len(trajs))
			expert_traj = trajs[idx]
		self.track = expert_traj['observation'].copy()
		self.t = 0
		self.track_t = len(self.track)
		return np.concatenate([self.pose, self.track[self.t]], axis=0)

	def get_expert_trajs(self):
		return self.dataset.get_trajs()

#IDEA: also train an initial pose network?
if __name__=="__main__":
    env = HipHop()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = env.action_space.sample()
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print(len(s))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break