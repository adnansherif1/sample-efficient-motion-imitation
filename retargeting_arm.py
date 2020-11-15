import pickle
import json
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pytransform3d.rotations import *
import pytransform3d.transformations as p3dtr
import cv2

smpl_to_dm_map = np.array([0, 6, 12, 1, 4, 7, 16, 18, 20, 2, 5, 8, 17, 19, 21, 15, 10, 11])
left_arm = [12,13,14]
right_arm = [6,7,8]
map_body = [0,1,2,3,4,5,9,10,11]
externalities = [15,16,17] # meant to stabilize toes and head
one_d_rotations = [4, 7, 10, 13]
elbows = [7, 13]

vid_path = 'human_dynamics/demo_data'
vid = 'penn_action-2278.mp4'

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if np.all(vec1 == -vec2):
        return -np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if np.isnan(s):
        return np.eye(3)
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

def batch_global_rigid_transformation(Rs, Js, parent):
    N = Rs.shape[0]
    root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = np.expand_dims(Js, -1)

    def make_A(R, t):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]])
        t_homo = np.concatenate([t, np.ones([N, 1, 1])], 1)
        return np.concatenate([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, len(parent)):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = np.matmul(results[parent[i]], A_here)
        results.append(res_here)

    # N x 24 x 4 x 4
    results = np.stack(results, axis=1)

    new_J = results[:, :, :3, 3]

    Js_w0 = np.concatenate([Js, np.zeros([N, len(parent), 1, 1])], 2)
    init_bone = np.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = np.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
    A = results - init_bone

    return new_J, A

def get_R(points):
    """ Related to https://dspace.mit.edu/bitstream/handle/1721.1/6611/AIM-1378.pdf"""
    p0 = points[0]
    p01 = points[1] - points[0]
    p02 = points[2] - points[0]
    u = p01/np.linalg.norm(p01)
    v = p02 - np.dot(p02, u)*u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    w /= np.linalg.norm(w)
    return np.array([u, v, w]).T

def inv_kin(root_orient, model_coords, pose_coords, parents, terminal):
    """ Solve for the local joint rotations (parametrized as rotation matrices) 
        that transform model_coords into pose_coords .
    :param root_orient: A 3x3 root rotation/global pose orientation
    :param model_coords: An Nx3 numpy matrix with the 3d coordinate of each default joint
    :param pose_coords: An Nx3 numpy matrix with the 3d coordinate of each posed joint
    :param parents: A size N array giving the parent joint of each joint i
    :param terminal: A list of terminal joints

    :return Rs: An Nx3x3 numpy array of rotation matrices. 
    """
    N = model_coords.shape[0]
    Rs = [root_orient]
    As = [root_orient]

    # to solve for each rotation matrix, we have to see how it affects the orientation of the child joint(s)
    for i in range(1, N):
        if i in terminal: # the rotations of terminal joints don't actually matter b/c they have no children
            Rs.append(np.random.randn(3,3))
            As.append(np.random.randn(3,3))
        elif parents.count(i) > 1:
            children = [j for j, x in enumerate(parents) if x == i]

            J_here = model_coords[children] - model_coords[i]
            Jp_here = (np.linalg.inv(As[parents[i]])@(pose_coords[children] - pose_coords[i]).T).T
            H = J_here@Jp_here.T
            u, s, vh = np.linalg.svd(H)
            R_here = vh.T@u.T
            if np.linalg.det(R_here) < 0:
                # u, s, vh = np.linalg.svd(R_here)
                vh[2] *= -1
                R_here = vh.T@u.T

            Rs.append(R_here)
            As.append(As[parents[i]]@R_here)
        else:
            child = parents.index(i)
            j_here = model_coords[child] - model_coords[i]
            jp_here = np.linalg.inv(As[parents[i]])@(pose_coords[child] - pose_coords[i])
            R_here = rotation_matrix_from_vectors(j_here, jp_here)
            Rs.append(R_here)
            As.append(As[parents[i]]@R_here)

    return np.array([Rs]), np.array([As])

# ------------------------------------------------------------------------------------------------------------------

# load data
with open('DeepMimic/data/characters/humanoid3d.txt') as f:
    data = json.load(f)

with open('human_dynamics/demo_output/' + vid.split('.')[0] + '/hmmr_output/hmmr_output.pkl', 'rb') as f:
    hmmr_data = pickle.load(f, encoding='latin1')

with open('human_dynamics/models/neutral_smpl_with_cocoplustoesankles_reg.pkl', 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')

# get SMPL model info and orient to DeepMimic
smpl_joints = hmmr_data['J']
euler = [0, 0, 0]
euler[1] = -np.pi/2
rot_to_dm = matrix_from_euler_xyz(euler)
smpl_joints = np.array([(rot_to_dm @ frame.T).T for frame in smpl_joints])
posed_smpl = hmmr_data['J_transformed']
kt = smpl_model['kintree_table']

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

dm_parents += [2, 5, 11]  # need the extra joints to stabilize ankle/head rotations

# first calculate root offset
corrected = posed_smpl.copy()
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

corrected[:, :, 1] += 1

# inverse kinematics experiment with SMPL (sanity check)
np.set_printoptions(suppress=True)


T = hmmr_data['poses'].shape[0]

head_coords = (smpl_joints[:, smpl_to_dm_map[15]] - smpl_joints[:, smpl_to_dm_map[dm_parents[15]]] + joint_coords[dm_parents[15]]).reshape(T, 1, 3)
rtoe_coords = (smpl_joints[:, smpl_to_dm_map[16]] - smpl_joints[:, smpl_to_dm_map[dm_parents[16]]] + joint_coords[dm_parents[16]]).reshape(T, 1, 3)
ltoe_coords = (smpl_joints[:, smpl_to_dm_map[17]] - smpl_joints[:, smpl_to_dm_map[dm_parents[17]]] + joint_coords[dm_parents[17]]).reshape(T, 1, 3)

tiled_joint_coords = np.tile(joint_coords.reshape(1, joint_coords.shape[0], 3), (T, 1, 1))
joint_coords_retargeting = np.concatenate([tiled_joint_coords, head_coords, rtoe_coords, ltoe_coords], axis=1)

euler = [0, np.pi/2, 0]
initial_rot_offset = matrix_from_euler_xyz(euler)
euler = [np.pi, 0, 0]
y_axis_flip = matrix_from_euler_xyz(euler)

quarts = []
terminal = [8,14,15,16,17]

check = []

for t in range(T):
    start_frame = joint_coords_retargeting[t] + (corrected[t, 0] - joint_coords_retargeting[t, 0]) # gotta make sure your local rotations are calculated w.r.t correct starting frame
    retargeted_Rs, retargeted_As = inv_kin(initial_rot_offset@hmmr_data['poses'][t,0], start_frame, corrected[t], dm_parents, terminal)

    left_arm_Rs = np.array([[retargeted_As[0, 12], retargeted_Rs[0, 13], retargeted_Rs[0, 14]]])

    quart = []
    for i in range(len(left_arm_Rs[0])):
        R = left_arm_Rs[0, i]

        if i == 0:
            q = quaternion_from_matrix(-R, strict_check=False)
            # print(q)
            quart.append(q)
        else:

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

            except ValueError as e: # strict check should only fail for the random rotation matrices of the terminal joints
                # if i not in terminal:
                if i != 2:
                    print("unexpected bad R at joint %d, frame %d" % (i, t))
                    print(e)

    quarts.append(np.concatenate(quart))
    
    joints, _ = batch_global_rigid_transformation(left_arm_Rs, np.array([start_frame[left_arm]]), [-1, 0, 1])
    check.append(joints[0])

cap = cv2.VideoCapture(vid_path + '/' + vid)

fps = cap.get(cv2.CAP_PROP_FPS)
spf = 1/fps
root_motion = corrected[i, 12]
with open('retargeted/' + vid.split('.')[0] + '.txt', 'w') as f:
    f.write('{\n')
    f.write('"Loop": "wrap",\n')
    f.write('"Frames":\n')
    f.write('[\n')
    for i in range(len(quarts)):
        frame = [spf] + list(root_motion) + list(quarts[i])
        f.write(str(frame))
        if i < len(quarts)-1:
            f.write(',')
        f.write("\n")
    f.write(']\n')
    f.write('}')

# --------------------------------------------------------------------
# check final output one last time

check = np.concatenate([check], axis=0)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# scatters = [ ax.scatter(posed_smpl[0, :, 0], posed_smpl[0, :, 1], posed_smpl[0,:, 2]) for i in range(posed_smpl.shape[0]) ]
scatter = ax.scatter(posed_smpl[0, :, 0]+1, posed_smpl[0, :, 1], posed_smpl[0,:, 2], c='green')
scatter2 = ax.scatter(check[0, :, 0], check[0, :, 1], check[0,:, 2], c='blue')

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data.shape[0]):
        scatters[i]._offsets3d = (data[iteration, :, 0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters

def update_scatter(itr, data, scatter, data2, scatter2):
    scatter._offsets3d = (data[itr, :, 0]+1, data[itr, :, 1], data[itr, :, 2])
    scatter2._offsets3d = (data2[itr, :, 0], data2[itr, :, 1], data2[itr, :, 2])
    # return scatter, scatter2
    return scatter2

anim = animation.FuncAnimation(fig, update_scatter, posed_smpl.shape[0], fargs=([posed_smpl, scatter, check, scatter2]),
                                   interval=50, blit=False)
plt.show()