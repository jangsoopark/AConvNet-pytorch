#!/bin/bash

WORKSPACE=

docker run --gpus all --rm -it -p 8888:8888 --mount type=bind,src=${WORKSPACE},dst=/workspace aconvnet-pytorch /bin/bash
