import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy 
import scipy.optimize as optim
import scipy.stats as stats
from scipy.spatial.transform import Rotation as R

def get_optimal_R(base, target):
    def RthenDist(v):
        rot = R.from_quat(v).as_matrix()
        estimate = rot@base
        return np.sqrt(np.sum( (estimate-target)**2 ))

    res = optim.minimize(RthenDist, np.array([0, 1, 0, 0]))
    v_opt = res['x']
    R_opt = R.from_quat(v_opt).as_matrix()
    return R_opt, v_opt, res['fun']

def quart_from_R(matrix): # just copied from scipy Rotation but only keeping the nontrivial case
    is_single = False
    if matrix.shape == (3, 3):
        matrix = matrix.reshape((1, 3, 3))
        is_single = True

    num_rotations = matrix.shape[0]

    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, 1:] = matrix.diagonal(axis1=1, axis2=2)
    decision_matrix[:, 0] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))
    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, 1] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, 2] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, 3] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 0] = matrix[ind, k, j] - matrix[ind, j, k]

    quat /= np.linalg.norm(quat, axis=1)[:, None]
    if is_single:
        return quat[0]
    else:
        return quat


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    if np.all(a == -b):
        return -np.eye(3)
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
            # Rs.append(np.random.randn(3,3))
            # As.append(np.random.randn(3,3))
            Rs.append(np.eye(3))
            As.append(np.eye(3))
        elif parents.count(i) > 1:
            children = [j for j, x in enumerate(parents) if x == i]

            J_here = model_coords[children] - model_coords[i]
            Jp_here = (np.linalg.inv(As[parents[i]])@(pose_coords[children] - pose_coords[i]).T).T
            # H = J_here@Jp_here.T
            # u, s, vh = np.linalg.svd(H)
            # R_here = vh.T@u.T
            # if np.linalg.det(R_here) < 0:
                # u, s, vh = np.linalg.svd(R_here)
                # vh[2] *= -1
                # R_here = vh.T@u.T
            # R_here, _, f_error = get_optimal_R(J_here, Jp_here)
            # if f_error > 1e-1:
                # print("Warning: high error encountered in shoulder/head estimation")
            child = children[0]
            j_here = model_coords[child] - model_coords[i]
            jp_here = np.linalg.inv(As[parents[i]])@(pose_coords[child] - pose_coords[i])
            R_here = rotation_matrix_from_vectors(j_here, jp_here)
            # compare_single(J_here, Jp_here)
            # compare_single((R_here@J_here.T).T, Jp_here)

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

def inv_kin2(root_orient, model_coords, pose_coords, parents, terminal):
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
            # Rs.append(np.random.randn(3,3))
            # As.append(np.random.randn(3,3))
            Rs.append(np.eye(3))
            As.append(np.eye(3))
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
            # R_here, _, f_error = get_optimal_R(J_here, Jp_here)
            # if f_error > 1e-1:
                # print("Warning: high error encountered in shoulder/head estimation")

            Rs.append(R_here)
            As.append(As[parents[i]]@R_here)
        else:
            child = parents.index(i)
            j_here = model_coords[child] - model_coords[i]
            jp_here = np.linalg.inv(As[parents[i]])@(pose_coords[child] - pose_coords[i])
            to_local_frame = rotation_matrix_from_vectors(j_here, np.array([0,1,0]))
            j_here = to_local_frame@j_here
            jp_here = to_local_frame@jp_here
            R_here = rotation_matrix_from_vectors(j_here, jp_here)
            Rs.append(R_here)
            As.append(As[parents[i]]@R_here)

    return np.array([Rs]), np.array([As])

def animate_single(pts, interval=30):
    assert(len(pts.shape) == 3)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    scatter = ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0,:, 2], c='green')

    def update_scatter(itr, data, scatter):
      scatter._offsets3d = (data[itr, :, 0], data[itr, :, 1], data[itr, :, 2])
      return scatter

    anim = animation.FuncAnimation(fig, update_scatter, pts.shape[0], fargs=([pts, scatter]),
                                       interval=interval, blit=False)
    plt.show()


def compare_animate(pts1, pts2, interval=30):
    assert(len(pts1.shape) == 3 and len(pts2.shape) == 3)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    scatter = ax.scatter(pts1[0, :, 0], pts1[0, :, 1], pts1[0,:, 2], c='green')
    scatter2 = ax.scatter(pts2[0, :, 0], pts2[0, :, 1], pts2[0,:, 2], c='blue')

    def update_scatter(itr, data, scatter, data2, scatter2):
      scatter._offsets3d = (data[itr, :, 0], data[itr, :, 1], data[itr, :, 2])
      scatter2._offsets3d = (data2[itr, :, 0], data2[itr, :, 1], data2[itr, :, 2])
      # return scatter, scatter2
      return scatter2

    anim = animation.FuncAnimation(fig, update_scatter, pts1.shape[0], fargs=([pts1, scatter, pts2, scatter2]),
                                       interval=interval, blit=False)
    plt.show()

def compare_single(pts1, pts2):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    scatter = ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], c='green')
    scatter2 = ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c='blue')
    plt.show()

def get_root_correction(code):
    root_adjust = np.eye(3)
    if code == 0: # body facing x+ or x- or y+ or y- with head facing z+
        print('received case 0')
        euler = [-np.pi/2, 0, 0] 
        root_adjust = R.from_rotvec(euler).as_matrix()
    elif code == 1: # body facing x+ or x- or y+ or y- with head facing z-
        print('received case 1')
        euler = [np.pi/2, 0, 0] 
        root_adjust = R.from_rotvec(euler).as_matrix()
    elif code == 2: # completely upside down
        print('received case 2')
        euler = [0, 0, np.pi] # axis of rotation is arbitrary, just need to get it back to vertical
        root_adjust = R.from_rotvec(euler).as_matrix()
    else:
        print('no need to rotate')

    return root_adjust