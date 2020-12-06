import os
import time
import cv2
from sac.models.sac_agent import SACAgent
from sac.envs.dance_rev.env import HipHop
from sac.infrastructure.utils import sample_trajectory
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def denormalize(points, H, W):
    scale = np.array([[H, W]])
    return (0.5 * (points + 1.0)) * scale

def draw_pose_openpose(kps, res, kintree):
    H, W = res
    kps = denormalize(kps, H, W)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    body_colors = cm.rainbow(np.linspace(0, 1, len(kintree)))
    for i in range(1, len(kintree)):
        p1 = (int(kps[i, 0]), int(kps[i, 1]))
        p2 = (int(kps[kintree[i], 0]), int(kps[kintree[i], 1]))
        color = tuple(body_colors[i][:3] * 255)
        img = cv2.line(img, p1, p2, color, 5)
        img = cv2.circle(img, p2, 6, color, -1)
    return img

def animate(obs, gt, kintree, res=(640, 640)):
    """
    obs -- (T, 50)
    gt -- (T, 50)
    kintree - (2, 25)
    """
    fig, ax = plt.subplots(1, 2)

    obs = obs.reshape(-1, 25, 2)
    gt = gt.reshape(-1, 25, 2)

    ims1 = [draw_pose_openpose(ob, res, kintree) for ob in obs]
    ims2 = [draw_pose_openpose(f, res, kintree) for f in gt]

    im1 = ax[0].imshow(ims1[0])
    im2 = ax[1].imshow(ims2[0])

    def init():
        im1.set_data(ims1[0])
        im2.set_data(ims2[0])
        return [im1, im2] 

    def animate(i):
        im1.set_data(ims1[i])
        im2.set_data(ims2[i])
        return [im1, im2]

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(ims1), interval=20, blit=True)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

def sample_w_gt(env, policy, max_path_length):
    # TODO: get this from hw1 or hw2
    ob = env.reset()
    obs = []
    steps = 0
    while True:
        obs.append(ob)
        ac = policy.get_action(ob, True)
        ob, rew, done, _ = env.step(ac)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            break
    gt_traj = env.dataset.get_trajs()[env.track_idx]['observation'][:, :env.dataset.ac_dim]
    return np.array(obs)[:, :env.dataset.ac_dim], gt_traj

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--exp_name', type=str, default='sanity')
    parser.add_argument('--load_epoch', type=int, default=50)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rb_size', type=int, default=1000000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=256)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--polyak', type=float, default=0.995)

    parser.add_argument('--clip_length', type=int, default=30)
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        print("experiment not found")

    # ------- load model -------
    env = HipHop()
    agent = SACAgent(env, params)
    # agent.load(logdir, args.load_epoch)

    kintree = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11]

    while True:
        traj, gt = sample_w_gt(env, agent, args.clip_length*args.fps)
        print(traj.shape)
        print(gt.shape)
        animate(traj, gt, kintree)

if __name__ == "__main__":
    main()
