import torch
from torch.utils.data import DataLoader
import logging
import os
import argparse
from datetime import datetime
import numpy as np

from FEEL.dataset.video_dataset import load_video_dataset, load_video
from FEEL.utils import timeit
from FEEL.model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController
# from save_and_load import load_model, save_model

CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
SIZE_EPISODE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DDHHMM = 242354 # 250233
NN = 24

model_paths = {
    "subcortical_pathway": f"/home/u01231/project_body/FEEL/outs/train_202411{DDHHMM}/subcortical_pathway_{NN}.pt",
    "pfc": f"/home/u01231/project_body/FEEL/outs/train_202411{DDHHMM}/pfc_{NN}.pt",
    "hippocampus": f"/home/u01231/project_body/FEEL/outs/train_202411{DDHHMM}/hippocampus_{NN}",
    "controller": f"/home/u01231/project_body/FEEL/outs/train_202411{DDHHMM}/controller_{NN}.pt"
}

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

def feel(video_path: str, model_paths: dict = model_paths) -> torch.Tensor:
    """
    Processes a video and outputs the emotional vector `eval2`.

    Args:
        video_path (str): Path to the video file.
        model_paths (dict): Dictionary containing paths to the pre-trained models:
            {
                "subcortical_pathway": str,
                "pfc": str,
                "hippocampus": str,
                "controller": str
            }

    Returns:
        torch.Tensor: The `eval2` emotional vector.
    """
    # Step 1: Prepare dataset
    model_mvit = EnhancedMViT(pretrained=True).to(device=DEVICE)
    
    characteristics = load_video(
        video_path = video_path,
        clip_length=CLIP_LENGTH,
        mvit=model_mvit,
    )
    # Step 2: Load models
    subcortical_pathway = SubcorticalPathway().to(DEVICE)
    pfc = PFC().to(DEVICE)
    controller = EvalController().to(DEVICE)

    subcortical_pathway.load_state_dict(torch.load(model_paths["subcortical_pathway"], map_location=DEVICE))
    pfc.load_state_dict(torch.load(model_paths["pfc"], map_location=DEVICE))
    controller.load_state_dict(torch.load(model_paths["controller"], map_location=DEVICE))
    hippocampus = HippocampusRefactored.load_from_file(model_paths["hippocampus"])
    
    # Step 3: Process video and generate `eval2`
    with torch.no_grad():    
        # Subcortical pathway for `eval1`
        eval1 = subcortical_pathway(characteristics.to(DEVICE))
        
        # Generate events and episodes via hippocampus
        events = hippocampus.receive(characteristics, eval1)
        if len(hippocampus) < hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (5, 1, 768))  # Adjust dimensions as needed
            pre_eval = pfc(episode)
        else:
            episode = hippocampus.generate_episodes_batch(events=events)
            pre_eval = pfc(episode.transpose(0, 1))
        
        # Final eval2 generation
        eval2 = controller(eval1, pre_eval)
        print(eval2.shape)
        return eval2.squeeze(0)