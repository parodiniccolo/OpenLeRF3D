ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=20.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}

# Install dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y
RUN apt install -y build-essential wget

#  for viewers
RUN apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
RUN apt install -y cmake git
# Install system dependencies
RUN apt-get update && \
    apt-get install -y -u git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
ENV CONDA_DIR /opt/conda
RUN echo "Installing conda..." \
	&& wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create --name=sai3d python=3.8.5

WORKDIR /media/dc-04-vol03/Niccolo/SAI3D

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "sai3d", "/bin/bash", "-c"]

# Activate Conda environment
RUN echo -e "#! /bin/bash\n\n# script to activate the conda environment" > ~/.bashrc \
    && conda init bash \
    && echo -e "\nconda activate sai3d" >> ~/.bashrc \
    && echo "echo \"Hello SAI3D\"" >>  ~/.bashrc \
    && conda clean -a

ENV BASH_ENV ~/.bashrc