#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

stamp=$(date +"%Y%m%d%H%M")
touch outs/$stamp.out
# python3 ECO_test.py > outs/$stamp.out
python3 runner.py > outs/$stamp.out