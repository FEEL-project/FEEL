#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
VIDEO_CACHE="dataset/splitted_TrainVal.json"

CMD_PARAMS="--video-cache $VIDEO_CACHE -o outs/res_$TIMESTAMP/ -m outs/res_2024-12-08_03-50-29/ -me 19"

echo "params: $CMD_PARAMS"
eval "poetry run python train.py $CMD_PARAMS"
echo "DONE"