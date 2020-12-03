import os
import sys
import pickle
import json
import numpy as np
from pytransform3d.rotations import *
import pytransform3d.transformations as p3dtr
import scipy
from retargeting_utils import *

h36m_to_dm_map = np.array([0, 7, 8, 1, 2, 3, 14, 15, 16, 4, 5, 6, 11, 12, 13, 9])
one_d_rotations = [4, 7, 10, 13]
elbows = [7,13]

def run(vid, viz=False):
    # ------------------------------------------------------------------------------------------------------------------
    # load data
    with open('DeepMimic/data/characters/humanoid3d.txt') as f:
        data = json.load(f)

    with open(vid, 'rb') as f:
        pred_3d = np.load(f)

    # get DeepMimic joints
    joints = data['Skeleton']['Joints']

    joint_coords = []
    dm_parents = []

    for joint_dict in joints:
        delta = np.array([joint_dict['AttachX'], joint_dict['AttachY'], joint_dict['AttachZ']])
        if joint_dict['Parent'] != -1:
            coords = joint_coords[joint_dict['Parent']] + delta
        else:
            coords = delta
        joint_coords.append(coords)
        dm_parents.append(joint_dict['Parent'])

    joint_coords = np.array(joint_coords)
    # ---------------------------------------------------retarget---------------------------------------------------------------

    corrected = pred_3d[:, h36m_to_dm_map]

    # then just adjust the relevant limb lengths
    adjusted = []
    for i in range(1, len(joint_coords)):
        limb_length = np.linalg.norm(joint_coords[i] - joint_coords[dm_parents[i]])
        current = corrected[:, i] - corrected[:, dm_parents[i]]
        adjusted.append(current/np.linalg.norm(current, axis=1).reshape(-1, 1) * limb_length)

    for i in range(1, len(joint_coords)):
        corrected[:, i] = corrected[:, dm_parents[i]] + adjusted[i-1]

    # normalize neck limb length (treat separately because not part of DM definition)
    neck_limb_length = np.mean(np.linalg.norm(pred_3d[:, 9] - pred_3d[:, 8], axis=1))
    current = corrected[:, -1] - corrected[:, 2]
    corrected[:, -1] = corrected[:, 2] + current/np.linalg.norm(current, axis=1).reshape(-1,1) * neck_limb_length 

    dm_parents += [2]  # need the extra joints to stabilize ankle/head rotations

    # correct base orientation of poses
    corrected[:, :, 1] = -corrected[:, :, 1] # need to flip y-axis for some reason, maybe cause image coords are reversed?

    np.set_printoptions(suppress=True)

    # ----------------------------------------------------visualize retarget--------------------------------------------------------------

    pred_3d[:, :, 1] = -pred_3d[:, :, 1]
    pred_3d[:, :, 0] += 1
    if viz:
        compare_animate(pred_3d, corrected)

    # ------------------------------------------------------get and save retargeted rotation matrices------------------------------------------------------------
    T = pred_3d.shape[0]

    head_coords = (np.array([0, neck_limb_length, 0]) + joint_coords[dm_parents[15]]).reshape(1,3)

    joint_coords_retargeting = np.concatenate([joint_coords, head_coords], axis=0)

    euler = [0, np.pi/2, 0]
    initial_rot_offset = matrix_from_euler_xyz(euler)

    start_frame = joint_coords_retargeting + (corrected[0, 0] - joint_coords_retargeting[0])
    start_frame = (initial_rot_offset@start_frame.T).T
    if viz:
        compare_single(start_frame, corrected[0])

    # root orient calc test
    root_orients = []
    children = [1,3,9]
    for t in range(T):
        start_frame = joint_coords_retargeting + (corrected[t, 0] - joint_coords_retargeting[0])
        start_frame = (initial_rot_offset@start_frame.T).T
        j_here = start_frame[children[0]] - start_frame[0]
        jp_here = (corrected[t, children[0]] - corrected[t, 0])

        R_opt = rotation_matrix_from_vectors(j_here, jp_here)

        # R_opt, q_opt, f_val = get_optimal_R(J_here.T, Jp_here.T)
        # if f_val > 1e-1:
            # print("Warning: high error root approx at frame %d" % t)
        root_orients.append(R_opt)

    root_approx = []
    for t in range(T):
        start_frame = joint_coords_retargeting + (corrected[t, 0] - joint_coords_retargeting[0])
        start_frame = (initial_rot_offset@start_frame.T).T
        root_approx.append(((root_orients[t]@start_frame.T).T)[children])
    root_approx = np.array(root_approx)

    if viz:
        compare_animate(corrected, root_approx)

    quarts = []
    terminal = [5,8,11,14,15]
    skip = [8,14,15]
    check = []

    for t in range(T):
        start_frame = joint_coords_retargeting[t] + (corrected[t, 0] - joint_coords_retargeting[t, 0]) # gotta make sure your local rotations are calculated w.r.t correct starting frame
        start_frame = (initial_rot_offset@start_frame.T).T
        retargeted_Rs, retargeted_As = inv_kin(root_orients[t], start_frame, corrected[t], dm_parents, terminal)
        quart = []
        for i in range(len(retargeted_Rs[0])):
            R = retargeted_Rs[0, i]
            if i in skip:
                continue
            try:
                q = quaternion_from_matrix(R)
                if i in one_d_rotations:
                    end_orient = np.linalg.inv(retargeted_As[0, i-1])@(corrected[t, i+1] - corrected[t, i])
                    start_orient = (start_frame[i+1] - start_frame[i])
                    if np.all(end_orient == -start_orient):
                        theta = np.pi
                    else:
                        theta = np.arccos(np.dot(end_orient, start_orient)/(np.linalg.norm(end_orient)*np.linalg.norm(start_orient))) # so far seems like a good enough approx
                        if i in elbows:  # intuitively, knees and elbows rotate w.r.t different axes
                            theta = 2*np.pi - theta

                    quart.append(np.array([theta]))
                else:
                    quart.append(q)

            except ValueError as e: # strict check shouldn't fail
                print("unexpected bad R at joint %d, frame %d" % (i, t))
                print(e)

        quarts.append(np.concatenate(quart))
        joints, _ = batch_global_rigid_transformation(retargeted_Rs, np.array([start_frame]), dm_parents)
        check.append(joints[0])

    fps = 30 # DanceRev says they use 30 fps
    spf = 1/fps

    with open(os.path.join('DeepMimic/data/motions/dancerev', vid.split('/')[-1].split('.')[0] + '.txt'), 'w') as f:
        f.write('{\n')
        f.write('"Loop": "wrap",\n')
        f.write('"Frames":\n')
        f.write('[\n')
        for i in range(len(quarts)):
            frame = [spf] + list(corrected[i,0]) + list(quarts[i])
            f.write(str(frame))
            if i < len(quarts)-1:
                f.write(',')
            f.write("\n")
        f.write(']\n')
        f.write('}')

    # --------------------------------------------------------------------
    # check final output one last time
    if viz:
        check = np.concatenate([check], axis=0)
        corrected[:, :, 0] += 1
        compare_animate(corrected, check)

if __name__ == "__main__":
    vid = 'VideoPose3D/output/0.npy'
    run(vid, True)

    # import glob as glob
    # all_vids = glob.glob('VideoPose3D/output/*')
    # for vid in all_vids:
        # run(vid)
