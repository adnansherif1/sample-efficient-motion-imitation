# deep-rl-dance

## Docker Setup
Make sure Docker is [installed](https://docs.docker.com/engine/install/) on your computer/instance then, run the following commands:
```
git clone REPO_LINK
git submodule init
git submodule update
docker build . -t deep-rl-dance
```
If this completes without errors, you now have a docker image ready to run training on. Example startup:
```
docker run -it deep-rl-dance
python mpi_run.py --arg_file args/train_humanoid3d_backflip_args.txt --num_workers 8
```
Training takes a while, so it is recommended to use tmux to keep the docker image up while you go about your day. If you get into errno=1 when running mpi, try ```export OMPI_MCA_btl_vader_single_copy_mechanism=none``` before running again as suggested [here](https://github.com/open-mpi/ompi/issues/4948). 

## Training Management
Make sure tmux is installed, run `tmux`, run the commands you need to set up training on docker, then detach from tmux using `CTRL+B d` (hold ctrl and b simultaneously and quickly pressing d). To go back into your tmux session, run `tmux attach -t 0`. 

You can also create multiple panes within tmux using `CTRL+B %`. Navigation is done using `CTRL+B ARROWKEY`.

## Finetuning
Look at `args/ft_ours_ags.txt` for the format. You basically just add a model checkpoint to a regular train file. Make sure all other params are the same as the base file used for training. 