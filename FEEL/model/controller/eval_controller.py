import torch
import torch.nn as nn

class EvalController(nn.Module):
    """
    Controller of Evaluation: balances the output of Subcortical Pathway and Prefrontal Cortex
    """
    in_dim: int
    out_dim: int
    
    def __init__(self, in_dim: int = 9, out_dim: int = 8):
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_stack = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        # self.linear_stack = nn.Sequential(
        #     nn.Linear(in_dim, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     nn.Linear(512, out_dim)
        # )
    
    def forward(self, primary: torch.Tensor, pref_cortex: torch.Tensor) -> torch.Tensor:
        """
        primary (evaluation1): output of Subcortical-Pathway 
        pref_cortex (pre-evaluation): output of prefrontal-cortex(前頭前野)
        """
        input = torch.cat((primary, pref_cortex), dim=1)
        out = self.linear_stack(input)
        return out