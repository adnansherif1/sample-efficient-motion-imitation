# deep-rl-dance

## Fresh Docker Setup
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
python mpi_run.py --arg_file args/train_humanoid3d_backflip_args.txt
```
Training takes a while, so it is recommended to use tmux to keep the docker image up while you go about your day.

## Fresh GCP Setup
Set up an instance following this [guide](https://github.com/cs231n/gcloud), then either do Docker setup on that instance or follow these instructions

```
sudo apt-get install -y libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev
sudo apt-get install -y mesa-utils
sudo apt-get install -y clang
sudo apt-get install -y cmake

sudo apt-get install -y libbullet-dev libbullet-extras-dev
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt-get install -y libglew-dev
sudo apt-get install -y build-essential libxmu-dev libxi-dev libgl-dev
sudo apt-get install -y swig
sudo apt-get install -y libopenmpi-dev

git clone REPO_LINK
git submodule init
git submodule update

cd DeepMimic
sudo python -m pip install -r requirements.txt
```
at this stage, check to make sure the python3 path in DeepMimicCore/Makefile is correct on this instance. If not, you have to find the correct path, which may be in /usr/include or /opt/anacadonda3. Any python3 version 3.6 and above should work, so this is flexible. For the lib path, it usually works to include 'm' at the end even if pythonXm doesn't actually exist.

If ```make python``` can't find -lGL, look in /usr/lib/x86_64-linux-gnu and check that libGL.so is a valid symlink (fix it if not).
