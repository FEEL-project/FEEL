#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

stamp=$(date +"%Y%m%d%H%M")

# コマンドライン引数を変数として定義
# DATA_DIR="/home/u01230/SoccerNarration/FEEL/data/small_data/test"

DATA_DIR="/home/u01230/SoccerNarration/FEEL/data/EmVidCap/Videos/EmVidCap-L/TrainVal_clips/splitted_Test"
ANNOTATION_PATH="/home/u01230/SoccerNarration/FEEL/annotation/params_test.csv"
OUT_DIR="/home/u01230/SoccerNarration/FEEL/outs/test_$stamp"
# DATA_DIR="/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted/trainval"
# ANNOTATION_PATH="/home/u01231/project_body/FEEL/annotation/joe/params_trainval.csv"
# OUT_DIR="/home/u01231/project_body/FEEL/outs/train_$stamp"
mkdir -p $OUT_DIR

# 変数を使用してコマンドを実行
poetry run python train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR --pfc_controller_train False > $OUT_DIR/err.out

# # 使用するモデルの重みファイルのパス
# SUBCORTICAL_PATHWAY="/home/u01231/project_body/FEEL/outs/train_202411230421/subcortical_pathway_10.pt"

# # 変数を使用してコマンドを実行
# python3 train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR --subcortical_pathway $SUBCORTICAL_PATHWAY --subcortical_pathway_train False --pfc_controller_train True > $OUT_DIR/err.out