import os
import time
from sac.infrastructure.sac_trainer import SAC_Trainer

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--ep_len', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sanity')
    parser.add_argument('--epochs', '-e', type=int, default=50)

    parser.add_argument('--batch_size', '-b', type=int, default=100) #steps to use for gradient update
    parser.add_argument('--eval_num_episodes', '-eb', type=int, default=10) 
    parser.add_argument('--steps_per_epoch', '-spe', type=int, default=4000)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rb_size', type=int, default=1000000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=256)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=10)

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--learning_starts', type=int, default=4000)
    parser.add_argument('--update_every', type=int, default=50)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

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

    trainer = SAC_Trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
