import torch
from torch.utils.data import DataLoader
import logging
import os
from datetime import datetime

from dataset.video_dataset import load_video_dataset
from utils import timeit
from model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController

torch.autograd.set_detect_anomaly(True) #FIXME: Remove me later

BATCH_SIZE = 1
CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
SIZE_EPISODE = 3
DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True

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
    return tensor

def train_models(
    data_loader: DataLoader,
    model_mvit: EnhancedMViT,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_subcortical_pathway: SubcorticalPathway,
    model_controller: EvalController,
    write_path: str = None
):
    """Trains the model
    - Do not train MViT (use preloaded model)
    - Train subcortical pathway on its own using given label
    - Train hippocampus to make pre_eval closer to eval2
    - Train controller to make eval2 closer to real
    Args:
        video_path (str): _description_
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if write_path is None:
        write_path = f"outs/train_{timestamp}"
    os.makedirs(write_path)
    logging.info(f"Training started at {timestamp}, writing to {write_path}")
    model_mvit.eval()
    model_pfc.train()
    model_subcortical_pathway.train()
    model_controller.train()
    
    loss_eval1 = torch.nn.MSELoss()
    loss_pfc = torch.nn.MSELoss()
    loss_controller = torch.nn.MSELoss()
    optim_eval1 = torch.optim.Adam(model_subcortical_pathway.parameters(), lr=0.001)
    params_eval2 = list(model_pfc.parameters()) + list(model_controller.parameters())
    optim_eval2 = torch.optim.Adam(params_eval2, lr=0.001)
    
    EPOCHS = 10
    
    
    for epoch in range(EPOCHS):
        # Per epoch
        # Reset state
        optim_eval1.zero_grad()
        optim_eval2.zero_grad()
        #FIXME: model_hippocampus等をリセットする必要ある？
        
        logging.info(f"Epoch {epoch}/{EPOCHS}")
        for i, data in enumerate(data_loader):
            #inputs, labels_eval1, labels_eval2 = data
            eval2 = torch.Tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            inputs, labels_eval2 = data
            labels_eval1 = torch.rand((1, 1)) #FIXME: eval1がないので適当に生成
            logging.debug(f"Iteration {i}")
            with torch.no_grad():
                _, characteristics, _ = model_mvit(inputs)
            # Infer eval1 (intuitive emotional response)
            out_eval1 = model_subcortical_pathway(characteristics.detach())
            # Infer eval2 (cognitive emotional response)
            events = model_hippocampus.receive(characteristics, out_eval1) #FIXME: ここout_eval1でいいの？
            for event in events:
                model_hippocampus.save_to_memory(event=event, eval1=out_eval1, eval2=eval2) #FIXME: eval2をどうするか
            if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
                # Not enough events to generate episode, use memory itself and zero-padding others
                episode = zero_padding(characteristics, (SIZE_EPISODE, BATCH_SIZE, DIM_CHARACTERISTICS))
                pre_eval = model_pfc(episode)
            else:
                # Generate episode and calculate pre_eval
                episode = model_hippocampus.generate_episodes_batch(events=events)
                pre_eval = model_pfc(episode.transpose(0, 1))
            # Calculate eval2 by controller
            out_eval2 = model_controller(out_eval1, pre_eval)
            # Train subcortical pathway
            loss1 = loss_eval1(out_eval1, labels_eval1)
            loss1.backward(retain_graph=True) #FIXME: retain_graph=Trueが必要？
            optim_eval1.step()
            # Train eval2
            loss_pfc = loss_pfc(pre_eval, labels_eval2)
            loss_controller = loss_controller(out_eval2, labels_eval2)
            combined_loss = loss_pfc + loss_controller
            combined_loss.backward()
            optim_eval2.step()
            logging.debug(f"Loss: {loss1}, {loss_pfc}, {loss_controller}")
        # Save model
        torch.save(model_subcortical_pathway.state_dict(), os.path.join(write_path, f"subcortical_pathway_{epoch}.pt"))
        torch.save(model_pfc.state_dict(), os.path.join(write_path, f"pfc_{epoch}.pt"))
        torch.save(model_controller.state_dict(), os.path.join(write_path, f"controller_{epoch}.pt"))
        model_hippocampus.save_to_file(os.path.join(write_path, f"hippocampus_{epoch}.json"))
        logging.info(f"Epoch {epoch} done")
    logging.info(f"Training finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            
            
            
            
        



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    train_loader = load_video_dataset("data/small_data/trainval", "annotation/params_trainval.csv", BATCH_SIZE, CLIP_LENGTH)
    model_mvit = EnhancedMViT(pretrained=True).to(device=DEVICE)
    model_pfc = PFC(DIM_CHARACTERISTICS, SIZE_EPISODE, 8).to(device=DEVICE)
    model_hippocampus = HippocampusRefactored(
        DIM_CHARACTERISTICS,
        SIZE_EPISODE,
        replay_rate=10,
        episode_per_replay=5,
        min_event_for_episode=5,
    )
    model_subcortical_pathway = SubcorticalPathway().to(device=DEVICE)
    model_controller = EvalController().to(device=DEVICE)
    train_models(
        train_loader,
        model_mvit,
        model_pfc,
        model_hippocampus,
        model_subcortical_pathway,
        model_controller
    )