import torch
from torch.utils.data import DataLoader
import logging
import os
import argparse
import json
from datetime import datetime
import numpy as np
from typing import Tuple
from dataset.video_dataset import load_video_dataset
from utils import timeit
from model import EnhancedMViT, PFC, HippocampusRefactored, SubcorticalPathway, EvalController
# from save_and_load import load_model, save_model

BATCH_SIZE = 20
CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
SIZE_EPISODE = 5
# DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_LOG_FREQ = 10
DEBUG = True

def eval2_to_eval1(eval2: torch.Tensor) -> torch.Tensor:
    """Convert eval2 to eval1
    Args:
        eval2 (torch.Tensor): Eval2 tensor
    Returns:
        torch.Tensor: Eval1 tensor
    """
    if eval2.size(1) != 8:
        raise ValueError(f"Invalid size of eval2: {eval2.size()}")
    if not isinstance(eval2, torch.Tensor):
        raise ValueError(f"eval2 is not a tensor: {eval2}")
    ret = ((eval2[:, 0]+eval2[:, 1])/2 - (eval2[:, 2]+eval2[:, 4]+eval2[:, 5]+eval2[:, 6])/4) * (2+eval2[:, 3]+eval2[:, 7]) / 4
    ret = ret.unsqueeze(1)
    return ret.to(DEVICE)

def zero_padding(data: torch.Tensor, size:tuple) -> torch.Tensor:
    """Zero padding to make data size to size
    Args:
        data (torch.Tensor): Data to pad
        size (int): Size to pad
    Returns:
        torch.Tensor: Padded data
    """
    tensor = torch.zeros(size)
    tensor[:data.size(0), :, :] = data
    return tensor.to(DEVICE)

def round_list(lst: list, digit: int = 3) -> list:
    """Round list to integer
    Args:
        lst (list): List to round
    Returns:
        list: Rounded list
    """
    return [f"%.{digit}f" % x for x in lst]

def test_models(
    data_loader: DataLoader,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_subcortical_pathway: SubcorticalPathway,
    model_controller: EvalController,
    add_to_hippocampus: bool = True
) -> Tuple[list, list]:
    """Trains the model
    - Do not train MViT (use preloaded model)
    - Train subcortical pathway on its own using given label
    - Train hippocampus to make pre_eval closer to eval2
    - Train controller to make eval2 closer to real
    Args:
        video_path (str): description
    """
    log_eval1 = []
    log_eval2 = []
    model_pfc.eval()
    model_subcortical_pathway.eval()
    model_controller.eval()
    
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.MSELoss()
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            logging.info(f"Batch {i}")
            characteristics, labels_eval2, name = data
            labels_eval1 = eval2_to_eval1(labels_eval2)
            logging.debug(f"labels2 {labels_eval2.tolist()}")
            out_eval1 = model_subcortical_pathway(characteristics)
            logging.debug(f"out_eval1 {out_eval1.tolist()}")
            events = model_hippocampus.receive(characteristics, out_eval1)
            if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
                episode = zero_padding(characteristics, (SIZE_EPISODE, out_eval1.shape[0], DIM_CHARACTERISTICS))
                pre_eval = model_pfc(episode)
            else:
                episode = model_hippocampus.generate_episodes_batch(events=events)
                pre_eval = model_pfc(episode.transpose(0, 1))
            out_eval2: torch.Tensor = model_controller(out_eval1, pre_eval)
            
            if add_to_hippocampus:
                for cnt, event in enumerate(events):
                    model_hippocampus.save_to_memory(event=event, eval1=out_eval1[cnt], eval2=out_eval2[cnt])
            # Log
            for j in range (len(name)):
                log_eval1.append({"name": name[j], "correct": labels_eval1[j].item(), "infer": out_eval1[j].item()})
                log_eval2.append({"name": name[j], "correct": labels_eval2[j].tolist(), "infer": out_eval2[j].tolist()})
                logging.getLogger("eval1").info(f"Name: {name[j]}, Correct: {labels_eval1[j].item()}, Infer: {out_eval1[j].item()}")
                logging.getLogger("eval2").info(f"Name: {name[j]}, Correct: {round_list(labels_eval2[j].tolist())}, Infer: {round_list(out_eval2[j].tolist())}")
            logging.getLogger("batch").info(f"Loss eval1: {loss1(out_eval1, labels_eval1)}, eval2: {loss2(out_eval2, labels_eval2)}")
            logging.getLogger("batch").info(f"Hippocampus: {len(model_hippocampus)} memories")
    return log_eval1, log_eval2
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-data", type=str, help="Path to testing data")
    parser.add_argument("--model-path", "-model", type=str, help="Path to model")
    parser.add_argument("--epoch", type=int, help="Number of epochs", default=49)
    parser.add_argument("--write-path", "-write", type=str, help="Path to write", default=f"outs/test_{datetime.now().strftime('%Y%m%d%H%M')}")
    parser.add_argument("--add-hippocampus", type=bool, help="Add to hippocampus", default=False)
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logging.error(f"Data path {args.data_path} does not exist")
        exit(1)
    
    if not os.path.exists(args.model_path):
        logging.error(f"Model path {args.model_path} does not exist")
        exit(1)
    
    if not os.path.exists(args.write_path):
        os.makedirs(args.write_path)
    else:
        logging.warning(f"Write path {args.write_path} already exists")
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, 
        format='{asctime} [{levelname:.4}] {name}: {message}', 
        style='{', 
        filename=f"{args.write_path}/logging.log",
        filemode='w'
    )
    index_epoch = args.epoch
    print(f"Saving to path {args.write_path}")
    logging.info(f"Start testing with the following parameters: {args}")
    
    data_loader = load_video_dataset(
        None,
        None,
        BATCH_SIZE,
        CLIP_LENGTH,
        EnhancedMViT(pretrained=True).to(device=DEVICE),
        cache_path=args.data_path
    )
    
    model_pfc = PFC(DIM_CHARACTERISTICS, SIZE_EPISODE).to(DEVICE)
    model_pfc.load_state_dict(torch.load(os.path.join(args.model_path, f"pfc_{index_epoch}.pt"), map_location=DEVICE))
    model_hippocampus = HippocampusRefactored.load_from_file(os.path.join(args.model_path, f"hippocampus_{index_epoch}"))
    model_subcortical_pathway = SubcorticalPathway().to(DEVICE)
    model_subcortical_pathway.load_state_dict(torch.load(os.path.join(args.model_path, f"subcortical_pathway_{index_epoch}.pt"), map_location=DEVICE))
    model_controller = EvalController().to(DEVICE)
    model_controller.load_state_dict(torch.load(os.path.join(args.model_path, f"controller_{index_epoch}.pt"), map_location=DEVICE))
    
    eval1, eval2 = test_models(
        data_loader,
        model_pfc,
        model_hippocampus,
        model_subcortical_pathway,
        model_controller,
        args.add_hippocampus
    )
    
    with open(f"{args.write_path}/eval1.json", "w") as f:
        json.dump(eval1, f, indent=4)
    with open(f"{args.write_path}/eval2.json", "w") as f:
        json.dump(eval2, f, indent=4)
    
    
    logging.info("Finished testing")