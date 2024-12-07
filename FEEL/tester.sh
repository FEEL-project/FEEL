#!/bin/sh

#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

stamp=$(date +"%Y%m%d%H%M")

# コマンドライン引数を変数として定義

# EmVidCap-Lを使用する場合
# DATA_DIR="/home/u01230/SoccerNarration/FEEL/data/EmVidCap/Videos/EmVidCap-L/TrainVal_clips/splitted_TrainVal"
# ANNOTATION_PATH="/home/u01230/SoccerNarration/FEEL/annotation/params_trainval.csv"
# VIDEO_CACHE="dataset/splitted_Test.json"
# OUT_DIR="outs/test_$stamp"

# joeを使用する場合
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

DATA_DIR="dataset/joe.json"
MODEL_DIR="outs/res_2024-12-08_03-50-29/"
OUT_DIR="outs/test_$TIMESTAMP"

# small_dataを使用する場合
# DATA_DIR="/home/u01231/project_body/FEEL/data/small_data/trainval"

# 出力ディレクトリが存在しない場合は作成
mkdir -p $OUT_DIR

# 変数を使用してコマンドを実行
# python3 train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR > $OUT_DIR/err.out
# python3 train_merged.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR > $OUT_DIR/err.out
# poetry run python train_merged.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR --video-cache $VIDEO_CACHE --replay true --epoch 6  > $OUT_DIR/err.out
# poetry run python train_merged.py --data_dir $DATA_DIR --video-cache $VIDEO_CACHE --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR --epoch $EPOCH --episode-size $EPISODE_SIZE > $OUT_DIR/err.out
COMMAND_PARAMS="-data $DATA_DIR -model $MODEL_DIR -write $OUT_DIR"
echo "params: $COMMAND_PARAMS"
poetry run python testing.py $COMMAND_PARAMS

# 使用したデータのパスなどをerr.outに書き込む
echo "DONE"

### 以上 ###