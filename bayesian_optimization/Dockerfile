# for A100s
FROM nvcr.io/nvidia/pytorch:22.02-py3 AS base

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

COPY environment.yml .

# Essentials
RUN apt update && \
    apt install -y build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    wget \
    make \
    g++ \
    libboost-all-dev \
    xutils-dev \
    libxss1 \
    xvfb \
    tmux && \
    rm -rf /var/lib/apt/lists


# Update pip
RUN SHA=ToUcHMe which python3
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install software required for pyscreener and vina based docking software
FROM base AS base-vina

RUN wget -O ADFRsuite.tar.gz https://ccsb.scripps.edu/adfr/download/1038/ \
    && mv ADFRsuite.tar.gz ../ && cd .. \
    && tar -xzvf ADFRsuite.tar.gz \
    && cd ADFRsuite_* \
    && echo "Y" | ./install.sh -d . -c 0 \
    && cd .. \
    && rm -rf ADFRsuite.tar.gz

ENV PATH="${PATH}:/ADFRsuite_x86_64Linux_1.0/bin:"

# Install smina
RUN wget -O smina https://sourceforge.net/projects/smina/files/smina.static/download \
    && chmod +x smina \
    && mv smina ../bin/

# mamba is way faster while installing
# See https://github.com/conda/conda/issues/8051#issuecomment-1549451621
RUN conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba

# Repo requirements
RUN conda env update -n base -f environment.yml

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local

# Change owner of home dir
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]
