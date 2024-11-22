# FEEL project

```sh
cd FEEL
```

## For Developers

We recommend that you follow the rules below.

If you want to do an experiment, you can run

```sh
python train.py --data_dir /path-to-data-directory/ --annotation_path /path-to-annotation-file/
```

However, our recommendation is that you edit or make trainer.sh like below.

```sh
stamp=$(date +"%Y%m%d%H%M")

# コマンドライン引数を変数として定義
DATA_DIR="/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted/trainval"
ANNOTATION_PATH="/home/u01231/project_body/FEEL/annotation/joe/params_trainval.csv"
OUT_DIR = "/home/u01231/project_body/FEEL/outs/train_$stamp"

# 変数を使用してコマンドを実行
python3 train.py --data-dir $DATA_DIR --annotation-path $ANNOTATION_PATH --out_dir $LOG_PATH > $OUT_DIR/err.out
```

Then, you will get some output files in the `outs/train_{timestamp}` directory. This directory is automatically made in the train_models / train_models_periods function.

- weights of Subcortical Pathway: `subcortical_pathway_{epoch}.pt`
- weights of Prefrontal Cortex: `pfc_{epoch}.pt` or `pfc_{period}_{epoch}.pt`
- information of Hippocampus: `hippocampus_{epoch}.json` or `hippocampus_{period}_{epoch}.json`
- weights of Controller: `controller_{epoch}.pt` or ``controller_{period}_{epoch}.pt`

If you train models additionally, 


