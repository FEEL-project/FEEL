import logging
import torch
from torch.utils.data import DataLoader
from utils import zero_padding, eval2_to_eval1
from model import SubcorticalPathway, PFC, HippocampusRefactored, EvalController

from typing_extensions import deprecated

from .config import TrainingConfig

LOGGER = logging.getLogger(__name__)

def train_subcortical_pathway_epoch(
    config: TrainingConfig,
    data_loader: DataLoader,
    model: SubcorticalPathway,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer,
):
    """Train subcortical pathway for one epoch

    Args:
        config (TrainingConfig): Training configuration
        data_loader (DataLoader): DataLoader for training
        model (SubcorticalPathway): Subcortical pathway model
        loss_fn (torch.nn.Module): Loss function
        optim (torch.optim.Optimizer): Optimizer
    
    Returns:
        None
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
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.info(f"Iteration {i}: loss {loss}")
    LOGGER.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_pre_eval_epoch(
    config: TrainingConfig,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    save_memory: bool,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer
) -> None:
    """Train pre_eval for one epoch

    Args:
        config (TrainingConfig): Training configuration
        data_loader (DataLoader): DataLoader for training
        model_subcortical_pathway (SubcorticalPathway): Subcortical pathway model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        save_memory (bool): Save memory to hippocampus
        loss_fn (torch.nn.Module): Loss function
        optim (torch.optim.Optimizer): Optimizer
    
    Returns:
        None
    """
    losses = []
    for i, data in enumerate(data_loader):
        characteristics, labels_eval2,_ = data
        characteristics = characteristics.to(config.device)
        labels_eval2 = labels_eval2.to(config.device)
        eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (config.episode_size, config.batch_size, config.dim_characteristics))
            print("Device:", episode.device)
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        loss = loss_fn(pre_eval, labels_eval2)
        loss.backward()
        optim.step()
        losses.append(loss)
        if save_memory:
            for cnt, event in enumerate(events):
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt])
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.info(f"Iteration {i}: loss {loss}")
    LOGGER.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_controller_epoch(
    config: TrainingConfig,
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
        config (TrainingConfig): Training configuration
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
            episode = zero_padding(characteristics, (config.episode_size, eval1.shape[0], config.dim_characteristics))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        out_eval2 = model_controller(eval1, pre_eval)
        loss = loss_fn(out_eval2, labels_eval2)
        loss.backward()
        optim.step()
        losses.append(loss)
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.info(f"Iteration {i}: loss {loss}")
    LOGGER.info(f"Average loss for epoch: {sum(losses)/len(losses)}")

def train_pfc_controller_epoch(
    config: TrainingConfig,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    save_memory: bool,
    model_controller: EvalController,
    loss_fn_label: torch.nn.Module,
    optim_pfc: torch.optim.Optimizer,
    loss_fn_pre: torch.nn.Module,
    optim_controller: torch.optim.Optimizer
) -> None:
    """Train PFC and controller for one epoch

    Args:
        config (TrainingConfig): Configuration
        data_loader (DataLoader): DataLoader for training
        model_subcortical_pathway (SubcorticalPathway): Subcortical pathway model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        save_memory (bool): Save memory to hippocampus
        model_controller (EvalController): Controller model
        loss_fn_label (torch.nn.Module): Loss function for controller
        optim_pfc (torch.optim.Optimizer): Optimizer for PFC
        loss_fn_pre (torch.nn.Module): Loss function for PFC
        optim_controller (torch.optim.Optimizer): Optimizer for controller
    """
    losses_2_to_label = []
    losses_2_to_pre = []
    for i, data in enumerate(data_loader):
        characteristics, labels_eval2, _ = data
        characteristics = characteristics.to(config.device)
        labels_eval2 = labels_eval2.to(config.device)
        
        eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (config.episode_size, eval1.shape[0], config.dim_characteristics))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        out_eval2 = model_controller(eval1, pre_eval)
        loss_2_to_label = loss_fn_label(out_eval2, labels_eval2)
        loss_2_to_pre = loss_fn_pre(out_eval2, pre_eval)
        total_loss = loss_2_to_label + loss_2_to_pre
        total_loss.backward()
        optim_pfc.step()
        optim_controller.step()
        losses_2_to_label.append(loss_2_to_label)
        losses_2_to_pre.append(loss_2_to_pre)
        if save_memory:
            for cnt, event in enumerate(events):
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt])
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.info(f"Iteration {i}: loss (eval2 to eval2_label) {loss_2_to_label}, loss (eval2 to pre_eval) {loss_2_to_pre}")
    LOGGER.info(f"Average loss for epoch (eval2 to eval2_label): {sum(losses_2_to_label)/len(losses_2_to_label)}")
    LOGGER.info(f"Average loss for epoch (eval2 to pre_eval): {sum(losses_2_to_pre)/len(losses_2_to_pre)}")     

def train_pfc_controller_epoch_contrast(
    config: TrainingConfig,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    save_memory: bool,
    model_controller: EvalController,
    loss_fn: torch.nn.Module,
    optim_pfc: torch.optim.Optimizer,
    optim_controller: torch.optim.Optimizer
) -> None:
    """Train PFC and controller for one epoch

    Args:
        config (TrainingConfig): Configuration
        data_loader (DataLoader): DataLoader for training
        model_subcortical_pathway (SubcorticalPathway): Subcortical pathway model
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        save_memory (bool): Save memory to hippocampus
        model_controller (EvalController): Controller model
        loss_fn (torch.nn.Module): Loss function for controller
        optim_pfc (torch.optim.Optimizer): Optimizer for PFC
        optim_controller (torch.optim.Optimizer): Optimizer for controller

    Returns:
        None
    """
    losses_2_to_label = []
    for i, data in enumerate(data_loader):
        characteristics, labels_eval2,_ = data
        characteristics = characteristics.to(config.device)
        labels_eval2 = labels_eval2.to(config.device)
        
        eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (config.episode_size, eval1.shape[0], config.dim_characteristics))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        out_eval2 = model_controller(eval1, pre_eval)
        loss_2_to_label = loss_fn(out_eval2, labels_eval2)
        loss_2_to_label.backward()
        optim_pfc.step()
        optim_controller.step()
        losses_2_to_label.append(loss_2_to_label)
        if save_memory:
            for cnt, event in enumerate(events):
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt]) 
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.info(f"Iteration {i}: loss (eval2 to eval2_label) {loss_2_to_label}")
    LOGGER.info(f"Average loss for epoch (eval2 to eval2_label): {sum(losses_2_to_label)/len(losses_2_to_label)}")

@deprecated("Does not work with the current implementation, use train_pfc_controller_epoch_with_replay instead")
def train_replay(
    config: TrainingConfig,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_controller: EvalController,
    loss_fn_label: torch.nn.Module,
    optim_pfc: torch.optim.Optimizer,
    loss_fn_pre: torch.nn.Module,
    optim_controller: torch.optim.Optimizer
) -> None:
    """Replay memories in hippocampus

    Args:
        config (TrainingConfig): Configuration
        model_pfc (PFC): PFC model
        model_hippocampus (HippocampusRefactored): Hippocampus model
        model_controller (EvalController): Controller model
        loss_fn_label (torch.nn.Module): Loss function for controller
        optim_pfc (torch.optim.Optimizer): Optimizer for PFC
        loss_fn_pre (torch.nn.Module): Loss function for PFC
        optim_controller (torch.optim.Optimizer): Optimizer for controller
    """
    losses_to_label = []
    losses_to_pre = []
    for i in range(config.replay_iteration):
        optim_pfc.zero_grad()
        optim_controller.zero_grad()
        events = model_hippocampus.replay(batch_size=config.batch_size)
        episode = model_hippocampus.generate_episodes_batch(events=events)
        eval1_replay = torch.stack([event.eval1 for event in events])  # eventsからeval1を取り出す
        labels_eval2 = torch.stack([event.eval2 for event in events])  # eventsからeval2を取り出す
        pre_eval2 = model_pfc(episode.transpose(0, 1))
        out_eval2_2 = model_controller(eval1_replay, pre_eval2)
        loss_to_label = loss_fn_label(out_eval2_2, labels_eval2)
        loss_to_pre = loss_fn_pre(pre_eval2, out_eval2_2)
        total_loss = loss_to_label + loss_to_pre
        total_loss.backward()
        optim_pfc.step()
        optim_controller.step()
        losses_to_label.append(loss_to_label)
        losses_to_pre.append(loss_to_pre)
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.debug(f"Replay: loss (eval2 to eval2_label) {loss_to_label}, loss (eval2 to pre_eval) {loss_to_pre}")
    LOGGER.info(f"Average loss for replay (eval2 to eval2_label): {sum(losses_to_label)/len(losses_to_label)}")
    LOGGER.info(f"Average loss for replay (eval2 to pre_eval): {sum(losses_to_pre)/len(losses_to_pre)}")
        

def train_pfc_controller_epoch_with_replay(
    config: TrainingConfig,
    data_loader: DataLoader,
    model_subcortical_pathway: SubcorticalPathway,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    epoch: int,
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
        characteristics, labels_eval2, _ = data
        characteristics = characteristics.to(config.device)
        labels_eval2 = labels_eval2.to(config.device)
        
        with torch.no_grad():
            eval1 = model_subcortical_pathway(characteristics)
        events = model_hippocampus.receive(characteristics, eval1)
        if len(model_hippocampus) < model_hippocampus.min_event_for_episode:
            episode = zero_padding(characteristics, (config.episode_size, eval1.shape[0], config.dim_characteristics))
            pre_eval = model_pfc(episode)
        else:
            episode = model_hippocampus.generate_episodes_batch(events=events)
            pre_eval = model_pfc(episode.transpose(0, 1))
        out_eval2 = model_controller(eval1, pre_eval)
        loss_2_to_label = loss_maximization(out_eval2, labels_eval2)
        loss_2_to_pre = loss_expectation(pre_eval, out_eval2)
        total_loss = loss_2_to_label + loss_2_to_pre
        total_loss.backward()
        optim_pfc.step()
        optim_controller.step()
        losses_2_to_label.append(loss_2_to_label)
        losses_2_to_pre.append(loss_2_to_pre)
        if epoch == 0:
            for cnt, event in enumerate(events):
                model_hippocampus.save_to_memory(event=event, eval1=eval1[cnt], eval2=labels_eval2[cnt])
        if config.log_frequency and i % config.log_frequency == 0:
            LOGGER.info(f"Iteration {i}: loss (eval2 to eval2_label) {loss_2_to_label}, loss (eval2 to pre_eval) {loss_2_to_pre}")
    if epoch % model_hippocampus.replay_rate == 0:
        for _ in range(config.replay_iteration):
            optim_pfc.zero_grad()
            optim_controller.zero_grad()
            events = model_hippocampus.replay(batch_size=config.batch_size)
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
        LOGGER.info(f"Replay: loss (eval2 to eval2_label) {loss_2_to_label}, loss (eval2 to pre_eval) {loss_2_to_pre}")
    LOGGER.info(f"Average loss for epoch {epoch} (eval2 to eval2_label): {sum(losses_2_to_label)/len(losses_2_to_label)}")
    LOGGER.info(f"Average loss for epoch {epoch} (eval2 to pre_eval): {sum(losses_2_to_pre)/len(losses_2_to_pre)}")
