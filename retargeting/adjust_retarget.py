import pickle
import json
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as Rot
from pytransform3d.rotations import quaternion_from_matrix
from retargeting_utils import *
import sys
from tqdm import tqdm
import os

og_path = 'MPI_HDM05/bk/'
vid_name = 'uar4_poses'
output_path = '../DeepMimic/data/motions/retargeted/'
fps = 30
spf = 1/fps

with open(os.path.join(output_path, vid_name+'.txt')) as f:
    retarget = json.load(f)

curr_frames = retarget['Frames']

root_adjust = np.array([0, -0.1,0]) # CHANGE THIS AS NEEDED

for i in tqdm(range(len(curr_frames))):
    new_root = np.array(curr_frames[i][1:4])+root_adjust
    curr_frames[i][1:4] = list(new_root)

retarget['Frames'] = curr_frames

with open(os.path.join(output_path, vid_name+'_1.txt'), 'w', encoding='utf-8') as f:
    json.dump(retarget, f, ensure_ascii=False, indent=4)