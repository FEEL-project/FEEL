import torch
import pickle
from model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController

def save_model(
    model_pfc: PFC = None,
    model_hippocampus: HippocampusRefactored = None,
    model_subcortical_pathway: SubcorticalPathway = None,
    model_controller: EvalController =  None,
    id: str = "default"
    ):
    """モデルを保存する関数"""
    if model_pfc is not None:
        torch.save(model_pfc.state_dict(), f"./weights/PFC_{id}.pth")
    if model_hippocampus is not None:
        model_hippocampus.save_to_file(f"./weights/hippocampus_{id}.pkl")
    if model_subcortical_pathway is not None:
        torch.save(model_subcortical_pathway.state_dict(), f"./weights/SCP_{id}.pth")
    if model_controller is not None:
        torch.save(model_controller.state_dict(), f"./weights/controller_{id}.pth")

def load_model(
    model_pfc: PFC = None,
    model_hippocampus: HippocampusRefactored = None,
    model_subcortical_pathway: SubcorticalPathway = None,
    model_controller: EvalController =  None,
    id: str = "default"
    ):
    """モデルをロードする関数"""
    if model_pfc is not None:
        model_pfc.load_state_dict(torch.load(f"./weights/PFC_{id}.pth", weights_only=True))
    if model_hippocampus is not None:
        model_hippocampus = HippocampusRefactored.load_from_file(f"./weights/hippocampus_{id}.pkl")
    if model_subcortical_pathway is not None:
        model_subcortical_pathway.load_state_dict(torch.load(f"./weights/SCP_{id}.pth", weights_only=True))
    if model_controller is not None:
        model_controller.load_state_dict(torch.load(f"./weights/controller_{id}.pth", weights_only=True))

