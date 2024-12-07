import torch

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
    return tensor.to(data.device)