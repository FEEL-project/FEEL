# FEEL project

紹介文

## Installation

1. download dataset

    Before start this, make sure that you are at the `FEEL` directory.

    [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset/tree/main)

    ```bash
    cd dataset

    git clone https://github.com/cvdfoundation/kinetics-dataset.git
    cd kinetics-dataset

    bash ./k400_downloader.sh   # wget takes about 9 hours

    bash ./k400_extractor.sh    # tar takes about 1 hours
    ```

    classify every video with annotation. (You are supposed to be at `FEEL` directory)

    ```py
    cd data/kinetics-dataset
    python3 arrange_by_classes.py ./k400
    ```

## Abbreviating Dataset loading process

Instead of loading video and converting the video to a 768-dimension vector every time the program is run (which takes a decent time), you can load the characteristic vectors from cache. Since the characteristic is calculatd from pre-trained model, this is stable.

1. In `dataset/video_dataset.py`, set `USE_DATASET_CACHE` to `True` to use this cache, and set the path by `VIDEO_DATASET_PATH`. Default is `/home/u01230/SoccerNarration/FEEL/dataset/video_dataset.json`.

## Usage

1. edit yaml configure

    open `config/XXXXX.yaml`, edit this.

2. execute below.

```sh
python3 runner.py --config config/random_input.yaml
```

### train

```sh
train_merged.py [-h] --data_dir DATA_DIR --annotation_path
                       ANNOTATION_PATH [--out_dir OUT_DIR]
                       [--subcortical_pathway SUBCORTICAL_PATHWAY]
                       [--hippocampus HIPPOCAMPUS] [--pfc PFC]
                       [--controller CONTROLLER]
                       [--subcortical_pathway_train SUBCORTICAL_PATHWAY_TRAIN]
                       [--pfc_controller_train PFC_CONTROLLER_TRAIN]
                       [--no-debug] [--no-video-cache]
                       [--log-frequency LOG_FREQUENCY]
                       [--batch-size BATCH_SIZE] [--epoch EPOCH]
                       [--episode-size EPISODE_SIZE]
                       [--video-cache VIDEO_CACHE]
```

-   `log-frequency` defines how often the log for batches should occur.
-   `batch-size` defines batch size
-   `epoch` defines number of epoch
-   `video-cache` defines the cache file of video
