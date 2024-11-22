import torch
from torch.utils.data import DataLoader
import logging
import os
from datetime import datetime

from dataset.video_dataset import load_video_dataset
from utils import timeit
from model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController
# from save_and_load import load_model, save_model

BATCH_SIZE = 1
CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
SIZE_EPISODE = 3
DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_LOG_FREQ = 10
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

def train_subcortical_pathway_epoch(
    data_loader: DataLoader,
    model_mvit: EnhancedMViT,
    model: SubcorticalPathway,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer
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
        inputs, _ = data
        label_eval1 = torch.rand((1, 1)) #FIXME: eval1がないので適当に生成
        with torch.no_grad():
            _, characteristics, _ = model_mvit(inputs)
        out_eval1 = model(characteristics)
        loss = loss_fn(out_eval1, label_eval1)
        loss.backward()
        optim.step()
        losses.append(loss)
        if i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").debug(f"Iteration {i}: loss {loss}")
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_pre_eval_epoch(
    data_loader: DataLoader,
    model_mvit: EnhancedMViT,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer
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
        inputs, labels_eval2 = data
        with torch.no_grad():
            _, characteristics, _ = model_mvit(inputs)
        eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        for event in events:
            model_hippocampus.save_to_memory(event=event, eval1=eval1, eval2=labels_eval2) #FIXME: eval2をどうするか。正解データを使うのはおかしい気がする
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (SIZE_EPISODE, BATCH_SIZE, DIM_CHARACTERISTICS))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        loss = loss_fn(pre_eval, labels_eval2)
        loss.backward()
        optim.step()
        losses.append(loss)
        if i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").debug(f"Iteration {i}: loss {loss}")
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_controller_epoch(
    data_loader: DataLoader,
    model_mvit: EnhancedMViT,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_controller: EvalController,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer
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
        inputs, labels_eval2 = data
        with torch.no_grad():
            _, characteristics, _ = model_mvit(inputs)
        eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        for event in events:
            # FIXME: 2回に分けて学習を進める場合、hippocampusの記憶管理はどうするか。重複して記憶されるのは問題ないか？
            model_hippocampus.save_to_memory(event=event, eval1=eval1, eval2=labels_eval2) #FIXME: eval2をどうするか。正解データを使うのはおかしい気がする
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (SIZE_EPISODE, BATCH_SIZE, DIM_CHARACTERISTICS))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        out_eval2 = model_controller(eval1, pre_eval)
        loss = loss_fn(out_eval2, labels_eval2)
        loss.backward()
        optim.step()
        losses.append(loss)
        if i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").debug(f"Iteration {i}: loss {loss}")
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

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
    
    EPOCHS = 10
    
    # First train subcortical pathway
    loss_eval1 = torch.nn.MSELoss()
    optim_eval1 = torch.optim.Adam(model_subcortical_pathway.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        optim_eval1.zero_grad()
        logging.getLogger("epoch").info(f"Epoch {epoch}/{EPOCHS}")
        train_subcortical_pathway_epoch(data_loader, model_mvit, model_subcortical_pathway, loss_eval1, optim_eval1)
        torch.save(model_subcortical_pathway.state_dict(), os.path.join(write_path, f"subcortical_pathway_{epoch}.pt"))
        logging.getLogger("epoch").info(f"Epoch {epoch} done")
    logging.info(f"Training Subcortical Pathway finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Then train pre_eval
    model_subcortical_pathway.eval()
    loss_pfc = torch.nn.MSELoss()
    optim_pre_eval = torch.optim.Adam(model_pfc.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        optim_pre_eval.zero_grad()
        logging.getLogger("epoch").info(f"Epoch {epoch}/{EPOCHS}")
        train_pre_eval_epoch(data_loader, model_mvit, model_pfc, model_hippocampus, loss_pfc, optim_pre_eval)
        torch.save(model_pfc.state_dict(), os.path.join(write_path, f"pfc_{epoch}.pt"))
        model_hippocampus.save_to_file(os.path.join(write_path, f"hippocampus_{epoch}.json"))
        logging.getLogger("epoch").info(f"Epoch {epoch} done, hippocampus has {len(model_hippocampus)} memories")
    logging.info(f"Training PFC finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Finally train controller
    model_pfc.eval()
    loss_controller = torch.nn.MSELoss()
    optim_controller = torch.optim.Adam(model_controller.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        optim_controller.zero_grad()
        logging.getLogger("epoch").info(f"Epoch {epoch}/{EPOCHS}")
        train_controller_epoch(
            data_loader,
            model_mvit,
            model_pfc,
            model_hippocampus,
            model_controller,
            loss_controller,
            optim_controller
        )
        torch.save(model_controller.state_dict(), os.path.join(write_path, f"controller_{epoch}.pt"))
        logging.getLogger("epoch").info(f"Epoch {epoch} done, hippocampus has {len(model_hippocampus)} memories")
    logging.info(f"Training controller finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='{asctime} [{levelname:.4}] {name}: {message}', style='{')
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("batch").setLevel(logging.DEBUG)
        logging.getLogger("epoch").setLevel(logging.DEBUG)
        
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("batch").setLevel(logging.INFO)
        logging.getLogger("epoch").setLevel(logging.INFO)
    
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
    # load_model(
    #     model_pfc=model_pfc,
    #     model_hippocampus=model_hippocampus,
    #     model_subcortical_pathway=model_subcortical_pathway,
    #     model_controller=model_controller,
    #     id="0"
    # )
    # model_hippocampus = HippocampusRefactored.load_from_file(f"./weights/hippocampus_{0}.pkl")
    train_models(
        train_loader,
        model_mvit,
        model_pfc,
        model_hippocampus,
        model_subcortical_pathway,
        model_controller
   )
    
    # save_model(
    #     model_pfc=model_pfc,
    #     model_hippocampus=model_hippocampus,
    #     model_subcortical_pathway=model_subcortical_pathway,
    #     model_controller=model_controller,
    #     id="0"
    # )