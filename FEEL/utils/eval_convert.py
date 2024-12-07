import torch

def eval2_to_eval1(eval2: torch.Tensor) -> torch.Tensor:
    """Convert eval2 to eval1
    Args:
        eval2 (torch.Tensor): Eval2 tensor
    Returns:
        torch.Tensor: Eval1 tensor
    """
    global DEVICE
    
    if eval2.size(1) != 8:
        raise ValueError(f"Invalid size of eval2: {eval2.size()}")
    if not isinstance(eval2, torch.Tensor):
        raise ValueError(f"eval2 is not a tensor: {eval2}")
    ret = ((eval2[:, 0]+eval2[:, 1])/2 - (eval2[:, 2]+eval2[:, 4]+eval2[:, 5]+eval2[:, 6])/4) * (2+eval2[:, 3]+eval2[:, 7]) / 4
    ret = ret.unsqueeze(1)
    return ret