#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

stamp=$(date +"%Y%m%d%H%M")

# コマンドライン引数を変数として定義
DATA_DIR="/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted/trainval"
ANNOTATION_PATH="/home/u01231/project_body/FEEL/annotation/joe/params_trainval.csv"
OUT_DIR="/home/u01231/project_body/FEEL/outs/train_$stamp"
mkdir -p $OUT_DIR

# 変数を使用してコマンドを実行
python3 train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR > $OUT_DIR/err.out