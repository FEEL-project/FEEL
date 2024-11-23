import torch
from torch.utils.data import DataLoader
import logging
import os
import argparse
from datetime import datetime

from dataset.video_dataset import load_video_dataset
from utils import timeit
from model import EnhancedMViT, PFC, HippocampusRefactored, SubcorticalPathway, EvalController

BATCH_SIZE = 20
CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
SIZE_EPISODE = 3
# DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ret = torch.tensor(((eval2[:, 0]+eval2[:, 1])/2 - (eval2[:, 2]+eval2[:, 4]+eval2[:, 5]+eval2[:, 6])/4) * (2+eval2[:, 3]+eval2[:, 7]) / 4)
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

def test_subcortical_pathway(
    data_loader: DataLoader,
    model: SubcorticalPathway,
    loss_fn: torch.nn.Module,
):
    """Train subcortical pathway for one epoch

    Args:
        data_loader (DataLoader): DataLoader for training
        model_mvit (EnhancedMViT): MViT model
        model (SubcorticalPathway): Subcortical pathway model
        loss_fn (torch.nn.Module): Loss function
        optim (torch.optim.Optimizer): Optimizer
    """
    losses = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            characteristics, label_eval2,_ = data
            label_eval1 = eval2_to_eval1(label_eval2)
            out_eval1 = model(characteristics)
            loss = loss_fn(out_eval1, label_eval1)
            losses.append(loss)
    logging.info(f"Average loss: {sum(losses)/len(losses)}")

def test_pre_eval(
    data_loader: DataLoader,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    loss_fn: torch.nn.Module,
):
    """Train pre_eval for one epoch

    Args:
        data_loader (DataLoader): DataLoader for training
        model_mvit (EnhancedMViT): MViT model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        model_controller (EvalController): Controller model
        loss_fn (torch.nn.Module): Loss function for PFC
        optim (torch.optim.Optimizer): Optimizer
    """
    losses = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            characteristics, labels_eval2,_ = data
            eval1 = model_subcortical_pathway(characteristics)
            events = model_hippocampus.receive(characteristics, eval1)
            if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
                episode = zero_padding(characteristics, (SIZE_EPISODE, BATCH_SIZE, DIM_CHARACTERISTICS))
                pre_eval = model_pfc(episode)
            else:
                episode = model_hippocampus.generate_episodes_batch(events=events)
                pre_eval = model_pfc(episode.transpose(0, 1))
            loss = loss_fn(pre_eval, labels_eval2)
            losses.append(loss)
        cnt = 0
        for event in events:
            model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt]) 
            cnt += 1
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def test_controller(
    data_loader: DataLoader,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_controller: EvalController,
    loss_fn: torch.nn.Module,
) -> None:
    """Train controller for one epoch

    Args:
        data_loader (DataLoader): DataLoader for training
        model_mvit (EnhancedMViT): MViT model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        model_controller (EvalController): Controller model
        loss_fn (torch.nn.Module): Loss function for controller
        optim (torch.optim.Optimizer): Optimizer

    Returns:
        None
    """
    losses = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            characteristics, labels_eval2,_ = data
            eval1 = model_subcortical_pathway(characteristics)
            events = model_hippocampus.receive(characteristics, eval1)
            if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
                episode = zero_padding(characteristics, (SIZE_EPISODE, eval1.shape[0], DIM_CHARACTERISTICS))
                pre_eval = model_pfc(episode)
            else:
                episode = model_hippocampus.generate_episodes_batch(events=events)
                pre_eval = model_pfc(episode.transpose(0, 1))
            out_eval2 = model_controller(eval1, pre_eval)
            loss = loss_fn(out_eval2, labels_eval2)
            losses.append(loss)
    logging.info(f"Average loss: {sum(losses)/len(losses)}")


def test_models (
    data_loader: DataLoader,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_subcortical_pathway: SubcorticalPathway,
    model_controller: EvalController,
    write_path: str = None,
    subcortical_pathway_train: bool = True,
    pfc_controller_train: bool = True,
):
    """Tests the model
    - Train subcortical pathway on its own using given label
    - Train controller and pre_eval in periods
        - Train hippocampus to make pre_eval closer to eval2
        - Train controller to make eval2 closer to real
    Args:
        video_path (str): _description_
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if write_path is None:
        write_path = f"outs/test_{timestamp}"
    os.makedirs(write_path, exist_ok=True)
    logging.info(f"Testing started at {timestamp}, writing to {write_path}")
    logging.info(f"Device is {DEVICE}")
    model_pfc.eval()
    model_subcortical_pathway.eval()
    model_controller.eval()
    
    
    # First train subcortical pathway
    loss_eval1 = torch.nn.MSELoss()
    logging.getLogger("epoch").info("evaluating subcortial")
    test_subcortical_pathway(data_loader, model_subcortical_pathway, loss_eval1)
    logging.info("Testing Subcortical Pathway finished")
    
    # Then train pre_eval and controller in periods
    if not pfc_controller_train:
        return
    # Train pre_eval
    loss_pfc = torch.nn.MSELoss()
    logging.getLogger("epoch").info("testing pfc")
    test_pre_eval(data_loader, model_pfc, model_hippocampus, loss_pfc)
    
    # Finally train controller
    loss_controller = torch.nn.MSELoss()
    logging.getLogger("epoch").info("testing controller")
    test_controller(
        data_loader,
        model_pfc,
        model_hippocampus,
        model_controller,
        loss_controller,
    )
    logging.info(f"Testing controller finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")



if __name__ == "__main__":
    # set data-path and annotation-path
    parser = argparse.ArgumentParser(description="Train a video model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the annotation file')
    parser.add_argument('--out_dir', type=str, required=False, help='Path to the output directory', default=None)
    parser.add_argument('--subcortical_pathway', type=str, required=False, help='Path to the subcortical pathway model', default=None)
    parser.add_argument('--hippocampus', type=str, required=False, help='Path to the hippocampus model', default=None)
    parser.add_argument('--pfc', type=str, required=False, help='Path to the prefrontal cortex model', default=None)
    parser.add_argument('--controller', type=str, required=False, help='Path to the controller model', default=None)
    parser.add_argument('--subcortical_pathway_train', type=bool, required=False, help='Train the subcortical pathway', default=True)
    parser.add_argument('--pfc_controller_train', type=bool, required=False, help='Train the PFC and controller', default=True)

    # 引数を解析
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(level=logging.WARNING, 
                        format='{asctime} [{levelname:.4}] {name}: {message}', 
                        style='{', 
                        filename=args.out_dir + f"/test_{timestamp}.log",
                        filemode='w')
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("batch").setLevel(logging.DEBUG)
        logging.getLogger("epoch").setLevel(logging.DEBUG)
        
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("batch").setLevel(logging.INFO)
        logging.getLogger("epoch").setLevel(logging.INFO)

    model_mvit = EnhancedMViT(pretrained=True).to(device=DEVICE)
    test_loader = load_video_dataset(args.data_dir, args.annotation_path, BATCH_SIZE, CLIP_LENGTH, model_mvit)
    model_pfc = PFC(DIM_CHARACTERISTICS, SIZE_EPISODE, 8).to(device=DEVICE)
    if args.pfc is not None:
        model_pfc.load_state_dict(torch.load(args.pfc, map_location=DEVICE))
    
    if args.hippocampus is not None:
        model_hippocampus = HippocampusRefactored.load_from_file(args.hippocampus)
    else:
        model_hippocampus = HippocampusRefactored(
            DIM_CHARACTERISTICS,
            SIZE_EPISODE,
            replay_rate=10,
            episode_per_replay=5,
            min_event_for_episode=5,
        )
    
    model_subcortical_pathway = SubcorticalPathway().to(device=DEVICE)
    if args.subcortical_pathway is not None:
        model_subcortical_pathway.load_state_dict(torch.load(args.subcortical_pathway, map_location=DEVICE))
    
    model_controller = EvalController().to(device=DEVICE)
    if args.controller is not None:
        model_controller.load_state_dict(torch.load(args.controller, map_location=DEVICE))

    
    test_models(
        test_loader,
        model_pfc,
        model_hippocampus,
        model_subcortical_pathway,
        model_controller,
        write_path=args.out_dir,
        subcortical_pathway_train=args.subcortical_pathway_train,
        pfc_controller_train=args.pfc_controller_train
    )
    