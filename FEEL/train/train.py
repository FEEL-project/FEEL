import logging
import torch
from torch.utils.data import DataLoader
from model import SubcorticalPathway, PFC, HippocampusRefactored, EvalController
import os
from datetime import datetime

from .config import TrainingConfig
from .epoch import train_subcortical_pathway_epoch, train_pfc_controller_epoch, train_pfc_controller_epoch_contrast, train_replay, train_pfc_controller_epoch_with_replay
from model import SubcorticalPathway, PFC, HippocampusRefactored, EvalController, EnhancedMViT
from dataset.video_dataset import load_video_dataset
LOGGER = logging.getLogger(__name__)


def train_models(
    config: TrainingConfig,
    data_loader: DataLoader,
    model_pfc: PFC,
    model_hippocampus: HippocampusRefactored,
    model_subcortical_pathway: SubcorticalPathway,
    model_controller: EvalController,
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
    LOGGER.info(f"Training started at {timestamp}, writing to {config.out_path}")
    LOGGER.info(f"Device is {config.device}")
    model_pfc.train()
    model_subcortical_pathway.train()
    model_controller.train()
    
    
    # First train subcortical pathway
    if config.subcortical_pathway_train:
        loss_eval1 = torch.nn.MSELoss()
        optim_eval1 = torch.optim.Adam(model_subcortical_pathway.parameters(), lr=0.001)
        for epoch in range(config.epochs):
            optim_eval1.zero_grad()
            LOGGER.info(f"Epoch {epoch}/{config.epochs}")
            train_subcortical_pathway_epoch(config, data_loader, model_subcortical_pathway, loss_eval1, optim_eval1)
            torch.save(model_subcortical_pathway.state_dict(), os.path.join(config.out_path, f"subcortical_pathway_{epoch}.pt"))
            LOGGER.info(f"Epoch {epoch} done")
        LOGGER.info(f"Training Subcortical Pathway finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not config.pfc_controller_train:
        return
    # Then train pre_eval
    model_subcortical_pathway.eval()
    loss_pfc = torch.nn.MSELoss()
    optim_pre_eval = torch.optim.Adam(model_pfc.parameters(), lr=0.001)
    loss_controller = torch.nn.MSELoss()
    optim_controller = torch.optim.Adam(model_controller.parameters(), lr=0.001)
    for epoch in range(config.epochs):
        optim_pre_eval.zero_grad()
        optim_controller.zero_grad()
        LOGGER.info(f"Epoch {epoch}/{config.epochs}")
        if config.contrast:
            train_pfc_controller_epoch_contrast(
                config,
                data_loader,
                model_subcortical_pathway,
                model_pfc,
                model_hippocampus,
                epoch==0,
                model_controller,
                loss_pfc,
                optim_pre_eval,
                optim_controller
            )
        elif config.replay:
            train_pfc_controller_epoch_with_replay(
                config,
                data_loader,
                model_subcortical_pathway,
                model_pfc,
                model_hippocampus,
                epoch,
                model_controller,
                loss_pfc,
                optim_pre_eval,
                loss_controller,
                optim_controller
            )
        else:
            train_pfc_controller_epoch(
                config,
                data_loader,
                model_subcortical_pathway,
                model_pfc,
                model_hippocampus,
                epoch==0,
                model_controller,
                loss_pfc,
                optim_pre_eval,
                loss_controller,
                optim_controller
            )
            # Replay
            # if config.replay and epoch % config.replay_iteration == 0:
            #     train_replay(
            #         config,
            #         model_pfc,
            #         model_hippocampus,
            #         model_controller,
            #         loss_pfc,
            #         optim_pre_eval,
            #         loss_controller,
            #         optim_controller
            #     )
            
        torch.save(model_pfc.state_dict(), os.path.join(config.out_path, f"pfc_{epoch}.pt"))
        torch.save(model_controller.state_dict(), os.path.join(config.out_path, f"controller_{epoch}.pt"))
        model_hippocampus.save_to_file(os.path.join(config.out_path, f"hippocampus_{epoch}.json"))
        LOGGER.info(f"Epoch {epoch} done, hippocampus has {len(model_hippocampus)} memories")
    LOGGER.info(f"Training PFC and Controller finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")

def train_all(config: TrainingConfig):
    model_mvit = EnhancedMViT(True).to(config.device)
    train_loader = load_video_dataset(
        video_dir=config.data_path,
        label_path=config.annotation_path,
        batch_size=config.batch_size,
        clip_length=config.clip_length,
        mvit=model_mvit,
        use_cache=config.data_cache_path,
        cache_path=config.data_cache_path
    )
    model_pfc = PFC(
        config.dim_characteristics,
        config.episode_size,
    ).to(config.device)
    
    if config.pfc_path:
        model_pfc.load_state_dict(torch.load(config.pfc_path, map_location=config.device))
    
    model_hippocampus: HippocampusRefactored = None
    if config.hippocampus_path:
        model_hippocampus = HippocampusRefactored.load_from_file(config.hippocampus_path, config.device)
    else:
        model_hippocampus = HippocampusRefactored(
            config.dim_characteristics,
            config.episode_size,
            replay_rate=config.epochs//config.replay_iteration,
            min_event_for_episode=config.min_event_for_episode,
            min_event_for_replay=config.min_event_for_replay
        )
    
    model_subcortical_pathway = SubcorticalPathway().to(config.device)
    if config.subcortical_pathway_path:
        model_subcortical_pathway.load_state_dict(torch.load(config.subcortical_pathway_path, map_location=config.device))
    
    model_controller = EvalController().to(config.device)
    if config.controller_path:
        model_controller.load_state_dict(torch.load(config.controller_path, map_location=config.device))
    
    train_models(
        config,
        train_loader,
        model_pfc,
        model_hippocampus,
        model_subcortical_pathway,
        model_controller
    )