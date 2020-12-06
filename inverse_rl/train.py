from trainer import AIRL_Trainer 
import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='DanceRev-HH-v0')
    parser.add_argument('--ep_len', type=int, default=24*30) #24fps * 30 seconds
    parser.add_argument('--exp_name', type=str, default='sanity')
    parser.add_argument('--iterations', '-itr', type=int, default=50)
    parser.add_argument('--eval_num_episodes', '-eb', type=int, default=10) 
    
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=10)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--save_every', type=int, default=5)

    # SAC params
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--start_steps', type=int, default=10000) # used to specify how many TOTAL env steps to take random actions (i.e. independent of policy update iteration)
    parser.add_argument('--learning_starts', type=int, default=4000) # used to offset training for a few env steps EACH policy update iteration
    parser.add_argument('--update_every', type=int, default=50) # used to specify how often to take gradient steps EACH policy update iteration
    parser.add_argument('--rb_size', type=int, default=1000000)
    parser.add_argument('--steps_per_epoch', '-spe', type=int, default=4000)  # now this refers to how many env steps to take EACH policy update iteration
    parser.add_argument('--p_batch_size', '-pb', type=int, default=100) #steps to use for gradient update
    parser.add_argument('--p_learning_rate', '-plr', type=float, default=1e-3)
    parser.add_argument('--p_n_layers', '-pl', type=int, default=2)
    parser.add_argument('--p_size', '-ps', type=int, default=256)

    # IRL params
    parser.add_argument('--irl_wt', type=float, default=1.0)
    parser.add_argument('--irl_batch_size', '-ib', type=int, default=100) #steps to use for gradient update
    parser.add_argument('--irl_learning_rate', '-ilr', type=float, default=1e-3)
    parser.add_argument('--irl_n_layers', '-il', type=int, default=2)
    parser.add_argument('--irl_size', '-is', type=int, default=256)
    parser.add_argument('--irl_steps_per_iter', '-ispi', type=int, default=50)
    parser.add_argument('--score_discriminator', action='store_true', default=False)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runlogs')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = AIRL_Trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
