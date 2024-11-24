import torch
from torch.utils.data import DataLoader
import logging
import os
import argparse
from datetime import datetime

from dataset.video_dataset import load_video_dataset
from utils import timeit
from model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController, event_data
# from save_and_load import load_model, save_model

BATCH_SIZE = 20
CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
# DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True
SIZE_EPISODE = 3
REPLAY_ITERATION = 10
BATCH_SIZE = 20
BATCH_LOG_FREQ = 10
EPOCHS = 10

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

def train_subcortical_pathway_epoch(
    data_loader: DataLoader,
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
        characteristics, label_eval2,_ = data
        label_eval1 = eval2_to_eval1(label_eval2)
        out_eval1 = model(characteristics)
        loss = loss_fn(out_eval1, label_eval1)
        loss.backward()
        optim.step()
        losses.append(loss)
        if BATCH_LOG_FREQ and i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").info(f"Iteration {i}: loss {loss}")
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_pre_eval_epoch(
    epoch: int,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
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
        loss.backward()
        optim.step()
        losses.append(loss)
        if BATCH_LOG_FREQ and i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").info(f"Iteration {i}: loss {loss}")
        if epoch==0:
            cnt = 0
            for event in events:
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt]) 
                cnt += 1
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_controller_epoch(
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
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
        loss.backward()
        optim.step()
        losses.append(loss)
        if BATCH_LOG_FREQ and i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").info(f"Iteration {i}: loss {loss}")
    logging.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_pfc_controller_epoch(
    epoch: int,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_controller: EvalController,
    loss_maximization: torch.nn.Module,
    optim_pfc: torch.optim.Optimizer,
    loss_expectation: torch.nn.Module,
    optim_controller: torch.optim.Optimizer
) -> None:
    """Train PFC and controller for one epoch

    Args:
        data_loader (DataLoader): DataLoader for training
        model_mvit (EnhancedMViT): MViT model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        model_controller (EvalController): Controller model
        loss_eval2 (torch.nn.Module): Loss function for controller
        optim_pfc (torch.optim.Optimizer): Optimizer for PFC
        loss_pfc (torch.nn.Module): Loss function for PFC
        optim_controller (torch.optim.Optimizer): Optimizer for controller

    Returns:
        None
    """
    losses_2_to_label = []
    losses_2_to_pre = []
    for i, data in enumerate(data_loader):
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
        loss_2_to_label = loss_maximization(out_eval2, labels_eval2)
        loss_2_to_pre = loss_expectation(out_eval2, pre_eval)
        total_loss = loss_2_to_label + loss_2_to_pre
        total_loss.backward()
        # loss_2_to_label.backward(retain_graph=True)
        # loss_2_to_pre.backward(retain_graph=True)
        optim_pfc.step()
        optim_controller.step()
        losses_2_to_label.append(loss_2_to_label)
        losses_2_to_pre.append(loss_2_to_pre)
        if BATCH_LOG_FREQ and i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").info(f"Iteration {i}: loss (eval2 to eval2_label) {loss_2_to_label}, loss (eval2 to pre_eval) {loss_2_to_pre}")
        if epoch==0:
            cnt = 0
            for event in events:
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt]) 
                cnt += 1
    logging.info(f"Average loss for epoch (eval2 to eval2_label): {sum(losses_2_to_label)/len(losses_2_to_label)}")
    logging.info(f"Average loss for epoch (eval2 to pre_eval): {sum(losses_2_to_pre)/len(losses_2_to_pre)}")     


def train_pfc_controller_epoch_contrast(
    epoch: int,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_controller: EvalController,
    loss_maximization: torch.nn.Module,
    optim_pfc: torch.optim.Optimizer,
    loss_expectation: torch.nn.Module,
    optim_controller: torch.optim.Optimizer
) -> None:
    """Train PFC and controller for one epoch

    Args:
        data_loader (DataLoader): DataLoader for training
        model_mvit (EnhancedMViT): MViT model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        model_controller (EvalController): Controller model
        loss_eval2 (torch.nn.Module): Loss function for controller
        optim_pfc (torch.optim.Optimizer): Optimizer for PFC
        loss_pfc (torch.nn.Module): Loss function for PFC
        optim_controller (torch.optim.Optimizer): Optimizer for controller

    Returns:
        None
    """
    losses_2_to_label = []
    for i, data in enumerate(data_loader):
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
        loss_2_to_label = loss_maximization(out_eval2, labels_eval2)
        loss_2_to_label.backward()
        optim_pfc.step()
        optim_controller.step()
        losses_2_to_label.append(loss_2_to_label)
        if BATCH_LOG_FREQ and i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").info(f"Iteration {i}: loss (eval2 to eval2_label) {loss_2_to_label}")
        if epoch==0:
            cnt = 0
            for event in events:
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt]) 
                cnt += 1
    logging.info(f"Average loss for epoch (eval2 to eval2_label): {sum(losses_2_to_label)/len(losses_2_to_label)}")
    # logging.info(f"Average loss for epoch (eval2 to pre_eval): {sum(losses_2_to_pre)/len(losses_2_to_pre)}")   

def train_pfc_controller_epoch_with_replay(
    epoch: int,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_controller: EvalController,
    loss_maximization: torch.nn.Module,
    optim_pfc: torch.optim.Optimizer,
    loss_expectation: torch.nn.Module,
    optim_controller: torch.optim.Optimizer
) -> None:
    """Train PFC and controller for one epoch

    Args:
        data_loader (DataLoader): DataLoader for training
        model_mvit (EnhancedMViT): MViT model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        model_controller (EvalController): Controller model
        loss_eval2 (torch.nn.Module): Loss function for controller
        optim_pfc (torch.optim.Optimizer): Optimizer for PFC
        loss_pfc (torch.nn.Module): Loss function for PFC
        optim_controller (torch.optim.Optimizer): Optimizer for controller

    Returns:
        None
    """
    losses_2_to_label = []
    losses_2_to_pre = []
    for i, data in enumerate(data_loader):
        characteristics, labels_eval2,_ = data
        with torch.no_grad():
            eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (SIZE_EPISODE, eval1.shape[0], DIM_CHARACTERISTICS))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        out_eval2 = model_controller(eval1, pre_eval)
        for i in range(len(labels_eval2)):
            logging.debug(f"Eval2: {labels_eval2[i]}, Predicted: {out_eval2[i]}")
        loss_2_to_label = loss_maximization(out_eval2, labels_eval2)
        loss_2_to_pre = loss_expectation(pre_eval, out_eval2)
        total_loss = loss_2_to_label + loss_2_to_pre
        total_loss.backward()
        optim_pfc.step()
        optim_controller.step()
        losses_2_to_label.append(loss_2_to_label)
        losses_2_to_pre.append(loss_2_to_pre)
        if BATCH_LOG_FREQ and i % BATCH_LOG_FREQ == 0:
            logging.getLogger("batch").info(f"Iteration {i}: loss (eval2 to eval2_label) {loss_2_to_label}, loss (eval2 to pre_eval) {loss_2_to_pre}")
        if epoch==0:
            cnt = 0
            for event in events:
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt])
                cnt += 1
    if epoch % model_hippocampus.replay_rate == 0 and epoch > 0:
        for _ in range(REPLAY_ITERATION):
            optim_pfc.zero_grad()
            optim_controller.zero_grad()
            events = model_hippocampus.replay(batch_size=BATCH_SIZE)
            episode = model_hippocampus.generate_episodes_batch(events=events)
            eval1_replay = torch.stack([event.eval1 for event in events])  # eventsからeval1を取り出す
            labels_eval2 = torch.stack([event.eval2 for event in events])  # eventsからeval2を取り出す
            pre_eval2 = model_pfc(episode.transpose(0, 1))
            out_eval2_2 = model_controller(eval1_replay, pre_eval2)
            loss_2_to_label2 = loss_maximization(out_eval2_2, labels_eval2)
            loss_2_to_pre2 = loss_expectation(pre_eval2, out_eval2_2)
            total_loss2 = loss_2_to_label2 + loss_2_to_pre2
            total_loss2.backward()
            optim_pfc.step()
            optim_controller.step()
        logging.getLogger("batch").debug(f"Replay at epoch {epoch}: loss (eval2 to eval2_label) {loss_2_to_label}, loss (eval2 to pre_eval) {loss_2_to_pre}")
    logging.info(f"Average loss for epoch {epoch} (eval2 to eval2_label): {sum(losses_2_to_label)/len(losses_2_to_label)}")
    logging.info(f"Average loss for epoch {epoch} (eval2 to pre_eval): {sum(losses_2_to_pre)/len(losses_2_to_pre)}")

def train_models(
    data_loader: DataLoader,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_subcortical_pathway: SubcorticalPathway,
    model_controller: EvalController,
    write_path: str = None,
    subcortical_pathway_train: bool = True,
    pfc_controller_train: bool = True,
    replay: bool = False,
    contrast: bool = False
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
    os.makedirs(write_path, exist_ok=True)
    logging.info(f"Training started at {timestamp}, writing to {write_path}")
    logging.info(f"Device is {DEVICE}")
    model_pfc.train()
    model_subcortical_pathway.train()
    model_controller.train()
    
    
    # First train subcortical pathway
    if subcortical_pathway_train:
        loss_eval1 = torch.nn.MSELoss()
        optim_eval1 = torch.optim.Adam(model_subcortical_pathway.parameters(), lr=0.001)
        for epoch in range(EPOCHS):
            optim_eval1.zero_grad()
            logging.getLogger("epoch").info(f"Epoch {epoch}/{EPOCHS}")
            train_subcortical_pathway_epoch(data_loader, model_subcortical_pathway, loss_eval1, optim_eval1)
            torch.save(model_subcortical_pathway.state_dict(), os.path.join(write_path, f"subcortical_pathway_{epoch}.pt"))
            logging.getLogger("epoch").info(f"Epoch {epoch} done")
        logging.info(f"Training Subcortical Pathway finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not pfc_controller_train:
        return
    # Then train pre_eval
    model_subcortical_pathway.eval()
    loss_pfc = torch.nn.MSELoss()
    optim_pre_eval = torch.optim.Adam(model_pfc.parameters(), lr=0.001)
    loss_controller = torch.nn.MSELoss()
    optim_controller = torch.optim.Adam(model_controller.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        optim_pre_eval.zero_grad()
        optim_controller.zero_grad()
        logging.getLogger("epoch").info(f"Epoch {epoch}/{EPOCHS}")
        if replay:
            train_pfc_controller_epoch_with_replay(
                epoch,
                data_loader,
                model_subcortical_pathway,
                model_pfc,
                model_hippocampus,
                model_controller,
                loss_pfc,
                optim_pre_eval,
                loss_controller,
                optim_controller
            )
        elif contrast:
            train_pfc_controller_epoch_contrast(
                epoch,
                data_loader,
                model_subcortical_pathway,
                model_pfc,
                model_hippocampus,
                model_controller,
                loss_pfc,
                optim_pre_eval,
                loss_controller,
                optim_controller
            )
        else:
            train_pfc_controller_epoch(
                epoch,
                data_loader,
                model_subcortical_pathway,
                model_pfc,
                model_hippocampus,
                model_controller,
                loss_pfc,
                optim_pre_eval,
                loss_controller,
                optim_controller
            )
        torch.save(model_pfc.state_dict(), os.path.join(write_path, f"pfc_{epoch}.pt"))
        torch.save(model_controller.state_dict(), os.path.join(write_path, f"controller_{epoch}.pt"))
        model_hippocampus.save_to_file(os.path.join(write_path, f"hippocampus_{epoch}.json"))
        logging.getLogger("epoch").info(f"Epoch {epoch} done, hippocampus has {len(model_hippocampus)} memories")
    logging.info(f"Training PFC and Controller finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")

    
if __name__ == "__main__":
    # set data-path and annotation-path
    parser = argparse.ArgumentParser(description="Train a video model")
    parser.add_argument('--data_dir', type=str, required=False, help='Path to the dataset directory', default=None)
    parser.add_argument('--annotation_path', type=str, required=False, help='Path to the annotation file', default=None)
    parser.add_argument('--out_dir', type=str, required=False, help='Path to the output directory', default=None)
    parser.add_argument('--subcortical_pathway', type=str, required=False, help='Path to the subcortical pathway model', default=None)
    parser.add_argument('--hippocampus', type=str, required=False, help='Path to the hippocampus model', default=None)
    parser.add_argument('--pfc', type=str, required=False, help='Path to the prefrontal cortex model', default=None)
    parser.add_argument('--controller', type=str, required=False, help='Path to the controller model', default=None)
    parser.add_argument('--subcortical_pathway_train', type=bool, required=False, help='Train the subcortical pathway', default=True)
    parser.add_argument('--pfc_controller_train', type=bool, required=False, help='Train the PFC and controller', default=True)
    parser.add_argument('--replay', type=bool, required=False, help='Use replay in hippocampus', default=False)
    parser.add_argument('--contrast', type=bool, required=False, help='Use contrastive learning', default=False)
    parser.add_argument('--no-debug', action='store_false', help='Enable debug logging')
    parser.add_argument('--no-video-cache', action='store_false', help='Disable video cache')
    parser.add_argument('--log-frequency', type=int, required=False, help='Log frequency', default=10)
    parser.add_argument('--batch-size', type=int, required=False, help='Batch size', default=20)
    parser.add_argument('--epoch', type=int, required=False, help='Number of epochs to run', default=20)
    parser.add_argument('--episode-size', type=int, required=False, help='Number of epochs to run', default=3)
    parser.add_argument('--video-cache', type=str, required=False, help='Video cache', default=None)
    
    # 引数を解析
    args = parser.parse_args()
    DEBUG = args.no_debug
    BATCH_LOG_FREQ = args.log_frequency
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    SIZE_EPISODE = args.episode_size
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(level=logging.WARNING, 
                        format='{asctime} [{levelname:.4}] {name}: {message}', 
                        style='{', 
                        filename=args.out_dir + f"/train_{timestamp}.log",
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
    # train_loader = load_video_dataset("data/small_data/trainval", "annotation/params_trainval.csv", BATCH_SIZE, CLIP_LENGTH)
    train_loader = load_video_dataset(
        video_dir=args.data_dir,
        label_path=args.annotation_path,
        batch_size=BATCH_SIZE,
        clip_length=CLIP_LENGTH,
        mvit=model_mvit,
        use_cache=args.no_video_cache,
        cache_path=args.video_cache
    )
    model_pfc = PFC(DIM_CHARACTERISTICS, SIZE_EPISODE, 8).to(device=DEVICE)
    if args.pfc is not None:
        model_pfc.load_state_dict(torch.load(args.pfc, map_location=DEVICE))
    
    if args.hippocampus is not None:
        model_hippocampus = HippocampusRefactored.load_from_file(args.hippocampus)
    else:
        model_hippocampus = HippocampusRefactored(
            DIM_CHARACTERISTICS,
            SIZE_EPISODE,
            replay_rate=EPOCHS//REPLAY_ITERATION,
            min_event_for_episode=10,
            min_event_for_replay=20,
        )
    
    model_subcortical_pathway = SubcorticalPathway().to(device=DEVICE)
    if args.subcortical_pathway is not None:
        model_subcortical_pathway.load_state_dict(torch.load(args.subcortical_pathway, map_location=DEVICE))
    
    model_controller = EvalController().to(device=DEVICE)
    if args.controller is not None:
        model_controller.load_state_dict(torch.load(args.controller, map_location=DEVICE))
    
    train_models(
        train_loader,
        model_pfc,
        model_hippocampus,
        model_subcortical_pathway,
        model_controller,
        write_path=args.out_dir,
        subcortical_pathway_train=args.subcortical_pathway_train,
        pfc_controller_train=args.pfc_controller_train,
        replay=args.replay,
        contrast=args.contrast
    )
