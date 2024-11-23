#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

stamp=$(date +"%Y%m%d%H%M")

# コマンドライン引数を変数として定義

# EmVidCap-Lを使用する場合
# DATA_DIR="/home/u01230/SoccerNarration/FEEL/data/EmVidCap/Videos/EmVidCap-L/TrainVal_clips/splitted_Test"
# ANNOTATION_PATH="/home/u01230/SoccerNarration/FEEL/annotation/params_test.csv"
# OUT_DIR="/home/u01230/SoccerNarration/FEEL/outs/test_$stamp"

# joeを使用する場合
DATA_DIR="/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted/trainval"
ANNOTATION_PATH="/home/u01231/project_body/FEEL/annotation/joe/params_trainval.csv"
OUT_DIR="outs/train_$stamp"

# small_dataを使用する場合
# DATA_DIR="/home/u01231/project_body/FEEL/data/small_data/trainval"

# 出力ディレクトリが存在しない場合は作成
mkdir -p $OUT_DIR

# 変数を使用してコマンドを実行
poetry run python train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR > $OUT_DIR/err.out

# 使用したデータのパスなどをerr.outに書き込む
echo "DATA_DIR: $DATA_DIR" > $OUT_DIR/err.out
echo "ANNOTATION_PATH: $ANNOTATION_PATH" >> $OUT_DIR/err.out
echo "OUT_DIR: $OUT_DIR" >> $OUT_DIR/err.out

### 以上

# # 使用するモデルの重みファイルのパス
# SUBCORTICAL_PATHWAY="/home/u01231/project_body/FEEL/outs/train_202411230421/subcortical_pathway_10.pt"

# # 変数を使用してコマンドを実行
# python3 train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR --subcortical_pathway $SUBCORTICAL_PATHWAY --subcortical_pathway_train False --pfc_controller_train True > $OUT_DIR/err.out