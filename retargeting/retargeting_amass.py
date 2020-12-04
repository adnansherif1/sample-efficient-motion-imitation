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

smpl_to_dm_map = np.array([0, 6, 12, 2, 5, 8, 17, 19, 21, 1, 4, 7, 16, 18, 20, 15, 11, 10])
one_d_rotations = [4, 7, 10, 13]
elbows = [7, 13]
out_path = '../DeepMimic/data/motions/retargeted'

def run(vid, viz=False, all_case=None):
    # ------------------------------------------------------------------------------------------------------------------

    # load data
    vid_data = np.load(vid, allow_pickle=True)
    assert(vid_data['poses'].shape[-1] == 156), 'poses are of dim ' + str(len(vid_data['poses']))  # expecting SMPL-H format, otherwise retargeting isn't guaranteed to work
    data = vid_data['poses'][:, :66]
    T = data.shape[0]
    data = data.reshape(T, -1, 3)

    with open('../DeepMimic/data/characters/humanoid3d.txt') as f:
        dm_model = json.load(f)

    with open('../human_dynamics/models/neutral_smpl_with_cocoplustoesankles_reg.pkl', 'rb') as f:
        smpl_model = pickle.load(f, encoding='latin1')

    # get SMPL model info and orient to DeepMimic
    base = smpl_model['J'][:-2]
    smpl_parents = smpl_model['kintree_table'][0][:-2] # dont need last two palm joints

    # convert data to 3d pose
    Rs = np.array([Rot.from_rotvec(data[i]).as_matrix() for i in range(T)])

    posed, _ = batch_global_rigid_transformation(Rs, np.tile(np.expand_dims(base,0), (T, 1, 1)), smpl_parents)

    # get DeepMimic joints
    joints = dm_model['Skeleton']['Joints']

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

    euler = [0, -np.pi/2, 0]
    dm_to_smpl = Rot.from_rotvec(euler).as_matrix()
    euler = [0, np.pi/2, 0]
    smpl_to_dm = Rot.from_rotvec(euler).as_matrix()

    # sanity = joint_coords.copy()
    # sanity = np.concatenate([sanity, [[1,0,0]]], axis=0)

    # sanity = np.concatenate([[sanity[0]], (dm_to_smpl@sanity[1:].T).T], axis=0)

    # compare_single(sanity, base)

    # ---------------------------------------------------retarget---------------------------------------------------------------

    dm_parents += [2, 5, 11]  # need the extra joints to stabilize ankle/head rotations

    # first calculate root offset
    corrected = posed.copy()
    corrected[:, 0] = (corrected[:, 1] + corrected[:, 2])/2
    corrected = corrected[:, smpl_to_dm_map]

    # then just adjust the relevant limb lengths
    adjusted = []
    for i in range(1,15):
        limb_length = np.linalg.norm(joint_coords[i] - joint_coords[dm_parents[i]])
        current = corrected[:, i] - corrected[:, dm_parents[i]]
        adjusted.append(current/np.linalg.norm(current, axis=1).reshape(-1, 1) * limb_length)

    adjusted.append(corrected[:, 15] - corrected[:, dm_parents[15]])
    adjusted.append(corrected[:, 16] - corrected[:, dm_parents[16]])
    adjusted.append(corrected[:, 17] - corrected[:, dm_parents[17]])

    for i in range(1,18):
        corrected[:, i] = corrected[:, dm_parents[i]] + adjusted[i-1]

    # corrected[:, :, 1] += 1 # set character higher on y axis
    corrected = corrected - corrected[:, 0:1] + joint_coords[0:1]
    np.set_printoptions(suppress=True)

    # orientation correction
    if all_case is None:
        animate_single(posed)
        root_adjust = get_root_correction(int(input('which case? (0/1/2/any number)')))
    else:
        root_adjust = get_root_correction(all_case)

    # ------------------------------------------------- visualize retarget-----------------------------------------------------------------

    # posed[:, :, 0] += 1
    corrected2 = corrected.copy()
    corrected2 += np.expand_dims(vid_data['trans'],1) - np.expand_dims(corrected2[:, 0], 1)
    corrected2 = np.array([(root_adjust@corrected2[i].T).T for i in range(T)])
    posed += np.expand_dims(vid_data['trans'],1) - np.expand_dims(posed[:, 0], 1)
    if viz:
        compare_animate(posed, corrected2, interval=12)
    # sys.exit(0)
    # ------------------------------------------------------get and save retargeted rotation matrices-----------------------------------------------------------

    head_coords = (smpl_to_dm@(base[smpl_to_dm_map[15]] - base[smpl_to_dm_map[dm_parents[15]]] + dm_to_smpl@joint_coords[dm_parents[15]]).T).T.reshape(1, 3)
    rtoe_coords = (smpl_to_dm@(base[smpl_to_dm_map[16]] - base[smpl_to_dm_map[dm_parents[16]]] + dm_to_smpl@joint_coords[dm_parents[16]])).T.reshape(1, 3)
    ltoe_coords = (smpl_to_dm@(base[smpl_to_dm_map[17]] - base[smpl_to_dm_map[dm_parents[17]]] + dm_to_smpl@joint_coords[dm_parents[17]])).T.reshape(1, 3)

    joint_coords_retargeting = np.concatenate([joint_coords, head_coords, rtoe_coords, ltoe_coords], axis=0)
    # compare_single(sanity, joint_coords_retargeting)
    # sys.exit(0)

    corrected_body = np.array([(root_adjust@corrected[i, 1:].T).T for i in range(T)])
    corrected = np.concatenate([corrected[:, 0:1], corrected_body], axis=1)

    quarts = []
    terminal = [8,14,15,16,17]
    all_Rs = []

    # # TODO: can we batch this somehow?
    for t in tqdm(range(T), 'processing frames...'):
        root_orient = root_adjust@Rs[t, 0]@dm_to_smpl
        retargeted_Rs, retargeted_As = inv_kin(root_orient, joint_coords_retargeting, corrected[t], dm_parents, terminal) # retargeted_Rs = (1,22,3,3)
        quart = []
        for i in range(len(retargeted_Rs[0])):
            R = retargeted_Rs[0, i]

            if i == 0:
                q = quaternion_from_matrix(R, strict_check=False)
                quart.append(q)
            elif i not in terminal:
                try:
                    q = quaternion_from_matrix(R)
                    if i in one_d_rotations:

                        end_orient = np.linalg.inv(retargeted_As[0, dm_parents[i]])@(corrected[t, dm_parents.index(i)] - corrected[t, i])
                        start_orient = (joint_coords_retargeting[dm_parents.index(i)] - joint_coords_retargeting[i])
                        if np.all(end_orient == -start_orient):
                            theta = np.pi
                        else:
                            theta = np.arccos(np.dot(end_orient, start_orient)/(np.linalg.norm(end_orient)*np.linalg.norm(start_orient))) # so far seems like a good enough approx
                            if i not in elbows:  # intuitively, knees and elbows rotate w.r.t different axes
                                theta = 2*np.pi - theta

                        quart.append(np.array([theta]))
                    else:
                        quart.append(q)

                except ValueError as e: # strict check should only fail for the random rotation matrices of the terminal joints
                    print("unexpected bad R at joint %d, frame %d" % (i, t))
                    print(e)
        quarts.append(np.concatenate(quart))
        all_Rs.append(retargeted_Rs[0])

    fps = vid_data['mocap_framerate']
    spf = 1/fps

    root_motion = (root_adjust@vid_data['trans'].T).T
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    outfile = os.path.join(out_path, vid.split('/')[-1].split('.')[0] + '.txt')
    with open(outfile, 'w') as f:
        f.write('{\n')
        f.write('"Loop": "wrap",\n')
        f.write('"Frames":\n')
        f.write('[\n')
        for i in range(len(quarts)):
            frame = [spf] + list(root_motion[i]) + list(quarts[i])
            f.write(str(frame))
            if i < len(quarts)-1:
                f.write(',')
            f.write("\n")
        f.write(']\n')
        f.write('}')

    # ----------------------------------check final output one last time----------------------------------
    if viz:
        all_Rs = np.array(all_Rs)
        check = batch_global_rigid_transformation(all_Rs, np.tile(np.expand_dims(joint_coords_retargeting,0), (T, 1, 1)), dm_parents)[0]
        corrected[:, :, 0] += 1
        compare_animate(corrected, check)

if __name__ == "__main__":
    # vid = '../MPI_HDM05/mm/HDM_mm_08-01_01_120_poses.npz'
    # run(vid, True)

    import glob as glob
    all_vids = glob.glob('../MPI_Limits/*/*')
    for vid in all_vids:
        print("Processing %s" % vid)
        run(vid, all_case=0)
