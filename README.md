# FEEL project

## USAGE

```sh
python app.py
```

## For Developers

```sh
cd FEEL
```

We recommend that you follow the rules below.

If you want to do an experiment, you can run

```sh
python train.py --data_dir /path-to-data-directory/ --annotation_path /path-to-annotation-file/
```

However, our recommendation is that you edit or make trainer.sh like below.

```sh
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
```

Then, you will get some output files in the `outs/train_{timestamp}` directory. This directory is automatically made in the train_models / train_models_periods function.

- weights of Subcortical Pathway: `subcortical_pathway_{epoch}.pt`
- weights of Prefrontal Cortex: `pfc_{epoch}.pt` or `pfc_{period}_{epoch}.pt`
- information of Hippocampus: `hippocampus_{epoch}.json` or `hippocampus_{period}_{epoch}.json`
- weights of Controller: `controller_{epoch}.pt` or ``controller_{period}_{epoch}.pt`

If you train models additionally, the following arguments is to be added.

```sh
python train.py --data_dir /path-to-data-directory/ --annotation_path /path-to-annotation-file/ --subcortical_pathway /path-to-subcortical-pathway-weights/ --pfc /path-to-prefrontal-cortex-weights/ --hippocampus /path-to-hippocampus-information/ --controller /path-to-controller-weights/ --subcortical_pathway_train False --pfc_controller_train True
```

for example,

```sh
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

# 使用するモデルの重みファイルのパス
SUBCORTICAL_PATHWAY="/home/u01231/project_body/FEEL/outs/train_202411230421/subcortical_pathway_10.pt"

# 変数を使用してコマンドを実行
python3 train.py --data_dir $DATA_DIR --annotation_path $ANNOTATION_PATH --out_dir $OUT_DIR --subcortical_pathway $SUBCORTICAL_PATHWAY --subcortical_pathway_train False --pfc_controller_train True > $OUT_DIR/err.out
```
