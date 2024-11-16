# Kinetics - Downloader

## Usage
1. edit `dataset/downloader.sh`
2. move to `dataset` directory
3. run
      ```sh
      downloader.sh
      ```
## Directory Structure

```txt
FEEL/-+-data/-+-kinetics-400/-+-train/
      |       |               +-test/
      |       |               +-val/
      |       |
      |       +-kinetics-600/-+-train/
      |                       +-test/
      |                       +-val/
      |
      +-dataset/-+-downloader.sh
      |          +-converter.sh
      |          +-preprocessor.sh
      |          |
                 +ActivityNet/--Crawler/--Kinetics/-+-environment.yml
                 |                                  +-data/-+-kinetics-400_train.csv
                 |                                          +-kinetics-600_train.csv
                 |
```

## Usage (all in `downloader.sh`)
First, clone this repository and make sure that all the submodules are also cloned properly.
```
git clone https://github.com/activitynet/ActivityNet.git
cd ActivityNet/Crawler/Kinetics
```

Next, setup your environment
```
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl
```

Finally, download a dataset split by calling:
```
mkdir <data_dir>; python download.py {dataset_split}.csv <data_dir>
```