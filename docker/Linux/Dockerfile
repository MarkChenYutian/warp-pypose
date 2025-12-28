ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"
FROM ${BASE_IMAGE}

ARG  CUDA_MAJOR_VERSION="12"
ENV  DEBIAN_FRONTEND=noninteractive
ENV  CUDA_MAJOR_VERSION="${CUDA_MAJOR_VERSION}"

# Install additional development tools
# Dev    requirement: {tmux, libeigen3-dev}
RUN apt-get update && apt-get install -y \
    tmux libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup the working directory
WORKDIR /home/root/warp-workspace

# Freeze some key packages from the current environment
RUN  pip freeze | grep -E '^(torch==|torchvision==|numpy==|scipy==|tensorrt==)' \
     > /tmp/constraints.txt

# Load and install Python Requirements
COPY ./Material/requirements.txt /tmp/requirements.txt
RUN  pip install --no-cache-dir --upgrade-strategy only-if-needed\
     -r /tmp/requirements.txt -c /tmp/constraints.txt \
     && rm /tmp/requirements.txt

# Configure Git
RUN  git config --global --add safe.directory '*'

CMD ["bash"]
