import json
import glob
import numpy as np
import librosa
import os

pose_path='json/hiphop_1min_inter'
audio_path = 'audio/hiphop_1min'
compiled_path = 'compiled/hiphop_1min_inter'
outfilename = 'trajectories'

skip_pose = True

if not skip_pose:
	for traj in glob.glob(pose_path + '/*'):
		print("Compiling %s" % traj)
		kps_2d = []
		frames = glob.glob(traj + '/*')
		frames.sort(key=lambda x: int(x.split('/')[-1][7:11]))
		for frame in frames:
			with open(frame, 'rb') as f:
				data = json.load(f)
				kps_2d.append(data['people'][0]['pose_keypoints_2d'])
		
		full_trajectory = np.asarray(kps_2d)
		with open(compiled_path+'/'+traj.split('/')[-1]+'.npy', 'wb') as f:
			np.save(f, full_trajectory)

audio_files = glob.glob(audio_path+'/*')
base_names = [p.split('/')[-1].split(".")[0] for p in audio_files]

poses = []
pred_accuracies = []
audio = []
for base_name in base_names:
	print("Processing %s" % base_name)

	# load raw audio
	y, sr = librosa.load(os.path.join(audio_path, base_name+'.m4a'), sr=15360) # manually selected sampling rate to ensure audio and video frames are equal length/aligned

	# audio feature processing
	melspe = librosa.feature.melspectrogram(y=y, sr=sr)
	melspe_db = librosa.power_to_db(melspe, ref=np.max)
	mfcc = librosa.feature.mfcc(S=melspe_db)
	mfcc_delta = librosa.feature.delta(mfcc, width=3)

	audio_harmonic, audio_percussive = librosa.effects.hpss(y)
	chroma_cqt_harmonic = librosa.feature.chroma_cqt(y=audio_harmonic, sr=sr)

	onset_env = librosa.onset.onset_strength(audio_percussive, aggregate=np.median, sr=sr)
	tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
	onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
	beats_one_hot = np.zeros(len(onset_env))
	for idx in onset_beats:
		beats_one_hot[idx] = 1
	beats_one_hot = beats_one_hot.reshape(1, -1)
	onset_env = onset_env.reshape(1, -1)

	features = np.concatenate([
		mfcc,
		mfcc_delta,
		chroma_cqt_harmonic,
		onset_env,
		beats_one_hot,
		tempogram
	], axis=0)

	features = features.T # librosa gives you features as (feat_dim, n_frames)

	# load pose data
	with open(os.path.join(compiled_path, base_name+'.npy'), 'rb') as f:
		data = np.load(f)
	T = data.shape[0]
	data = data.reshape(T, 25, 3)
	pose_traj = data[:, :, :-1].reshape(T, -1)
	pred_accs = data[:, :, -1:].reshape(T, -1)

	# add 
	traj_length = min(T, features.shape[0])
	audio.append(features[:traj_length]) 
	poses.append(pose_traj[:traj_length])
	pred_accuracies.append(pred_accs[:traj_length])

assert(len(poses) == len(audio)), "not all audio/motion files have a unique pair"

audio_dim = audio[0].shape[-1]
obs_dim = audio_dim + poses[0].shape[-1]
print("Audio feature dim: %d" % audio_dim)
print("Obs dim: %d" % obs_dim)

acs = []
obs = []
nobs = []
trajs = []
for i in range(len(poses)):
	ac = poses[i][1:]-poses[i][:-1] 
	ob = np.concatenate([poses[i][:-1], audio[i][:-1]], axis=1)
	n_ob = np.concatenate([poses[i][1:], audio[i][1:]], axis=1)

	# acs.append(acs)
	# obs.append(ob)
	# nobs.append(nobs)
	trajs.append(np.concatenate([ob,ac,n_ob], axis=1))

p_min = np.min([np.min(pose) for pose in poses])
p_max = np.max([np.max(pose) for pose in poses])

pose_min = -np.ones(poses[0].shape[-1]) if p_min < 0 else np.zeros(poses[0].shape[-1])
pose_max = np.ones(poses[0].shape[-1]) if p_max > 2 else p_max*np.ones(poses[0].shape[-1])

# np.savez_compressed(outfilename, n_trajs=len(acs), acs=acs, obs=obs, nobs=nobs, obs_dim=obs_dim, audio_dim=audio_dim, pose_min=pose_min, pose_max=pose_max, pose_accuracies=pred_accuracies, num_kps=25) # tells downstream env to ignore 3rd keypoint dim or use it for weighted prediction accuracy
np.savez_compressed(outfilename, trajs=np.array(trajs, dtype=object), obs_dim=obs_dim, audio_dim=audio_dim, 
					pose_min=pose_min, pose_max=pose_max, pose_accuracies=np.array(pred_accuracies, dtype=object), num_kps=25) # tells downstream env to ignore 3rd keypoint dim or use it for weighted prediction accuracy

