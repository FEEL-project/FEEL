# FEEL project

紹介文

## Usage

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

2. convert video to images

    Before start this, make sure that you are at the `FEEL/dataset` directory
    ```sh
    # converter.sh # Legacy
    
    ```

3. execute test

    Before start this, make sure that you are at the `FEEL` directory
    ```sh
    sbatch execute.sh
    ```
