import argparse
import logging
import os

from train import TrainingConfig, train_all

def main(config: TrainingConfig):
    train_all(config)

def init_logger(path: str, debug: bool):
    """Initialize logger

    Args:
        path (str): Path to log file
        debug (bool): Enable debugging
    """
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(
        level=logging.WARNING, 
        format='{asctime} [{levelname:.4}] {name}: {message}', 
        style='{', 
        filename=path + f"/logger.log",
        filemode='w'
    )
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger("train").setLevel(logging.DEBUG if debug else logging.INFO)

if __name__ == "__main__":
    # set data-path and annotation-path
    parser = argparse.ArgumentParser(description="Train a video model")
    parser.add_argument('--data', "-d", type=str, required=False, help='Path to the dataset directory', default=None)
    parser.add_argument('--annotation', "-a", type=str, required=False, help='Path to the annotation file', default=None)
    parser.add_argument('--out', "-o", type=str, required=False, help='Path to the output directory', default="./outs/")
    parser.add_argument('--video-cache', type=str, required=False, help='Video cache', default=None)
    parser.add_argument('--epoch', "-e", type=int, required=False, help='Number of epochs to run', default=50)
    parser.add_argument('--batch-size', "-b", type=int, required=False, help='Batch size', default=20)
    parser.add_argument('--subcortical-pathway', type=str, required=False, help='Path to the subcortical pathway model', default=None)
    parser.add_argument('--subcortical-pathway-train', type=bool, required=False, help='Train the subcortical pathway', default=True)
    parser.add_argument('--hippocampus', type=str, required=False, help='Path to the hippocampus model', default=None)
    parser.add_argument('--pfc', type=str, required=False, help='Path to the prefrontal cortex model', default=None)
    parser.add_argument('--controller', type=str, required=False, help='Path to the controller model', default=None)
    parser.add_argument('--pfc-controller-train', type=bool, required=False, help='Train the PFC and controller', default=True)
    parser.add_argument('--replay', type=bool, required=False, help='Use replay in hippocampus', default=False)
    parser.add_argument('--contrast', type=bool, required=False, help='Use contrastive learning', default=False)
    parser.add_argument('--episode-size', type=int, required=False, help='Number of events an episode contains', default=5)
    parser.add_argument("--log-frequency", type=int, required=False, help="Log frequency", default=10)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    parser.add_argument("--model", "-m", type=str, required=False, help="Path to the model directory", default=None)
    parser.add_argument("--model-epoch", "-me", type=int, required=False, help="Epoch to load", default=None)
    
    # 引数を解析
    args = parser.parse_args()
    config = TrainingConfig.from_args(args)
    init_logger(args.out, args.debug)
    
    main(config)

    