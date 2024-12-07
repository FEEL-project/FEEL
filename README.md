# FEEL project

## Authors

-   Yuri Takigawa (03240425)
-   Kazuki Takahashi (03240501)
-   Satoshi Inoue (03240403)

## Dependencies

Using pyenv + poetry is recommended. The three of us were using 3.10.15 for development, and thus 3.10.15 is recommended.

Content of `pyproject.toml`

```
[tool.poetry]
name = "ai-exp-learning"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "~3.10"
gym = "0.22.0"
pyworld = "^0.3.4"
pysptk = "^1.0.1"
dt = "^1.1.65"
pyyaml = "^6.0.2"
numpy = "1.26.0"
torchvision = "0.14.1"
pygame = "^2.6.1"
transformers = "^4.46.2"
pandas = "^2.2.3"
opencv-python = "^4.10.0.84"
pathlib2 = "^2.3.7.post1"
datetime = "^5.5"
gradio = "^5.6.0"


[tool.poetry.group.dev.dependencies]
ipykernel = "^6.29.5"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## Implementation (overview)

The FEEL model consists of multiple sub-models, located in `model/` directory.

-   `EnhancedMViT` first processes the vision by converting the video to a 768-dimension vector, which holds the characteristics (特徴量) of the video. This is pre-trained, since the main target of our experiment is targeting the cognitive layers of the brain, while this `EnhancedMViT` model is closer to perception or sensation layer.
-   `SubcorticalPathway` is a fairly simple DNN model that takes in the 768-dimension characteristic vector as input and produces a "impulsive" or "quick" one-dimensional emotional response, in the range from -1 to 1.
-   Meanwhile, while the subcortical pathway produces a quick response, the `Hippocampus`, `Prefrontal Cortex (PFC)`, and the `Controller` is responsible for a more "cognitive" and slow response, where the brain recalls similar memories to enhance cognition.
-   The `Hippocampus` is essentially a memory storage unit that contains past memories, and given a new memory (a.k.a. `Event`) and enough memory stocked, returns similar memories and merges them together into a chunk of memories, organized in chronological order (a.k.a. `Episode`). The brain recognizes _"closer"_ memories as memories with similar characteristic vector (thus, the nearest memories).
-   The `Prefrontal Cortex (PFC)` takes in the `Episode`s generated in the `Hippocampus`, and generates a 8-dimensional "emotion" vector using a transformer model. We chose to use the transformer since it is known to excel at inferring relationships between the given data.
-   The `Controller` is responsible for adjusting the balance between the "impulsive", one-dimensional response produced through the `SubcorticalPathway` and the "cognitive", 8-dimensional response produced through `Hippocampus` and `PFC`.

## Usage

```sh
poetry run python train.py [-h] [--data DATA] [--annotation ANNOTATION] [--out OUT] [--video-cache VIDEO_CACHE] [--epoch EPOCH] [--batch-size BATCH_SIZE] [--subcortical-pathway SUBCORTICAL_PATHWAY]
                [--subcortical-pathway-train SUBCORTICAL_PATHWAY_TRAIN] [--hippocampus HIPPOCAMPUS] [--pfc PFC] [--controller CONTROLLER] [--pfc-controller-train PFC_CONTROLLER_TRAIN] [--replay REPLAY] [--contrast CONTRAST]
                [--episode-size EPISODE_SIZE] [--log-frequency LOG_FREQUENCY] [--debug] [--model MODEL] [--model-epoch MODEL_EPOCH]
```

or

```
sbatch train.sh
```

For running in IST with GPU.

```
options:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  Path to the dataset directory
  --annotation ANNOTATION, -a ANNOTATION
                        Path to the annotation file
  --out OUT, -o OUT     Path to the output directory
  --video-cache VIDEO_CACHE
                        Video cache
  --epoch EPOCH, -e EPOCH
                        Number of epochs to run
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --subcortical-pathway SUBCORTICAL_PATHWAY
                        Path to the subcortical pathway model
  --subcortical-pathway-train SUBCORTICAL_PATHWAY_TRAIN
                        Train the subcortical pathway
  --hippocampus HIPPOCAMPUS
                        Path to the hippocampus model
  --pfc PFC             Path to the prefrontal cortex model
  --controller CONTROLLER
                        Path to the controller model
  --pfc-controller-train PFC_CONTROLLER_TRAIN
                        Train the PFC and controller
  --replay REPLAY       Use replay in hippocampus
  --contrast CONTRAST   Use contrastive learning
  --episode-size EPISODE_SIZE
                        Number of events an episode contains
  --log-frequency LOG_FREQUENCY
                        Log frequency
  --debug               Enable debug logging
  --model MODEL, -m MODEL
                        Path to the model directory
  --model-epoch MODEL_EPOCH, -me MODEL_EPOCH
                        Epoch to load
```

-   `--data` and `--annotation` sets the path for the training data and its annotation.
-   `--video-cache` is a better alternative for `--data` and `--annotation`, and the video cache stores the 768-dimension vector produced via `EnhancedMViT`. Instead of processing 1768 of 10-second videos into 768-dimensional vectors, enabling the `--video-cache` option allows you to directly use the characteristic data (especially, since the `EnhancedMViT` model is pre-trained and no change will be made to the model). Four video caches are available, n the `dataset/` directory.

| Video    | Training dataset         | Testing dataset      |
| -------- | ------------------------ | -------------------- |
| EmVidCap | `splitted_TrainVal.json` | `splitted_Test.json` |
| Joe      | `joe.json`               | `joe_test.json`      |

-   `--out` specifies the output directory in which the trained models will be outputted.
-   `--subcortical-pathway`, `--hippocampus`, `--pfc`, and `--controller` arguments specify the corresponding models, if you are planning to resume learning on a already partially trained model.
-   `--model` and `--model-epoch` is an easier way to specify the partially trained model: specify the file directory in `--model` and the epoch number in `--model-epoch`, and the corresponding `SubcorticalPathway`, `Hippocampus`, etc. models will be loaded
-   Other settings are settings related to training behaviors.

### Examples

```sh
poetry run python train.py --video-cache dataset/splitted_TrainVal.json -o outs/res_2024-12-08_03-50-29/ --replay true -e 20
```

will run using EmVidCap's training data and output the models to `outs/res_2024-12-08_03-50-29/`. The replay learning is enabled, and 20 epochs will be run for training.

```sh
poetry run python train.py --video-cache dataset/splitted_TrainVal.json -o outs/res_2024-12-08_03-58-25/ -m outs/res_2024-12-08_03-50-29/ -me 19
```

This will use the 19th (20th) epoch model in the `outs/res_2024-12-08_03-50-29/` directory to further learn, and will output to `outs/res_2024-12-08_03-58-25/` directory.

### Other configurations

A more specific configuration is possible using the `train/config.py`'s `TrainingConfig` class. Configure the `TrainingConfig` as you wish, and just call `train/train.py`'s `train_all(config: TrainingConfig)` function.

## Testing

Running `testing.py` (not `test.py`) or `tester.sh` on `sbatch` will allow you to test, with the output being exported in JSON format with the correct value and inferred value.

Example

```sh
poetry run python testing.py -data dataset/joe.json -model outs/res_2024-12-08_03-50-29/ -write outs/test_2024-12-08_04-58-28
```
