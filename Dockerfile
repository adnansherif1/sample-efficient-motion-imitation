FROM ubuntu:18.04

RUN apt-get update && \
      apt-get -y install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

USER docker 

RUN echo "docker" | sudo -S apt update
 
RUN echo "docker" | sudo -S apt-get install -y python3.6
RUN echo "docker" | sudo -S apt-get install -y python3-pip

RUN echo "docker" | sudo -S apt-get install -y ssh

RUN echo "docker" | sudo -S apt-get install -y libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev
RUN echo "docker" | sudo -S apt-get install -y mesa-utils
RUN echo "docker" | sudo -S apt-get install -y clang
RUN echo "docker" | sudo -S apt-get install -y cmake

RUN echo "docker" | sudo -S apt-get install -y libbullet-dev libbullet-extras-dev
RUN echo "docker" | sudo -S apt-get install -y libeigen3-dev
RUN echo "docker" | sudo -S apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev
RUN echo "docker" | sudo -S apt-get install -y libglew-dev
RUN echo "docker" | sudo -S apt-get install -y build-essential libxmu-dev libxi-dev libgl-dev
RUN echo "docker" | sudo -S apt-get install -y swig
RUN echo "docker" | sudo -S apt-get install -y libopenmpi-dev

RUN echo "docker" | sudo -S pip3 install --no-cache-dir numpy PyOpenGL==3.1.0 PyOpenGL-accelerate==3.1.0 tensorflow mpi4py

WORKDIR /usr/src/deep-rl-dance
COPY . .

WORKDIR /usr/src/deep-rl-dance/DeepMimic/DeepMimicCore
RUN echo "docker" | sudo -S make python

WORKDIR /usr/src/deep-rl-dance/DeepMimic

USER root