from dataclasses import dataclass, field
import torch
import argparse
import os

@dataclass
class TrainingConfig:
    
    data_path: str|None = None
    annotation_path: str|None = None
    out_path: str|None = None
    data_cache_path: str|None = None
    epochs: int = 10
    batch_size: int = 20
    
    subcortical_pathway_path: str|None = None
    subcortical_pathway_train: bool = False
    
    hippocampus_path: str|None = None
    
    pfc_path: str|None = None
    controller_path: str|None = None
    pfc_controller_train: bool = False
    
    replay: bool = False
    replay_iteration: int = 5
    contrast: bool = False
    
    clip_length: int = 16
    dim_characteristics: int = 768
    episode_size: int = 3
    log_frequency: int = 10
    min_event_for_episode: int = 10
    min_event_for_replay: int = 20
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def set_model_paths(self, model_dir: str|None, epoch: int) -> None:
        if model_dir is None:
            return
        if not isinstance(epoch, int):
            raise ValueError("epoch must be an integer")
        self.subcortical_pathway_path = os.path.join(model_dir, f"subcortical_pathway_{epoch}.pt")
        self.hippocampus_path = os.path.join(model_dir, f"hippocampus_{epoch}.pt")
        self.pfc_path = os.path.join(model_dir, f"pfc_{epoch}.pt")
        self.controller_path = os.path.join(model_dir, f"controller_{epoch}.pt")
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        obj = cls()
        if args.data is not None:
            obj.data_path = args.data
        if args.annotation is not None:
            obj.annotation_path = args.annotation
        if args.out is not None:
            obj.out_path = args.out
        if args.video_cache is not None:
            obj.data_cache_path = args.video_cache
        if args.epoch is not None:
            obj.epochs = args.epoch
        if args.batch_size is not None:
            obj.batch_size = args.batch_size
        
        if args.subcortical_pathway is not None:
            obj.subcortical_pathway_path = args.subcortical_pathway
        if args.subcortical_pathway_train is not None:
            obj.subcortical_pathway_train = args.subcortical_pathway_train
        
        if args.hippocampus is not None:
            obj.hippocampus_path = args.hippocampus
        
        if args.pfc is not None:
            obj.pfc_path = args.pfc
        if args.controller is not None:
            obj.controller_path = args.controller
        if args.pfc_controller_train is not None:
            obj.pfc_controller_train = args.pfc_controller_train
        
        if args.replay is not None:
            obj.replay = args.replay
        if args.contrast is not None:
            obj.contrast = args.contrast
        if args.episode_size is not None:
            obj.episode_size = args.episode_size
        if args.log_frequency is not None:
            obj.log_frequency = args.log_frequency
        
        if args.model is not None and isinstance(args.model_epoch, int):
            obj.set_model_paths(args.model, args.model_epoch)
        
        return obj    