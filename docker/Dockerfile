
# docker build . -t aconvnet-pytorch
# Base container: docker pull pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN pip install seaborn && \
    pip install numpy && \
    pip install scipy&& \
    pip install tqdm && \
    pip install jupyter && \
    pip install matplotlib && \
    pip install scikit-image && \
    pip install scikit-learn && \
    pip install opencv-python && \
    pip install absl-py && \
    pip install optuna


RUN apt update && \
    apt install -y wget vim emacs nano libgl1-mesa-glx


RUN mkdir -p /workspace

ARG work_dir=/workspace

WORKDIR ${work_dir}
