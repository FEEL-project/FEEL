import torch
import torch.nn as nn

class PFC(nn.Module):
    """Prefrontal Cortex (前頭前野)
    Integration of Knowledge and Intelligence (知の統合)
    1. NeoCortex (新皮質): x (characteristics2) -> 512 knowledge
    2. DLPFC (背外側前頭前野): 512 knowledge -> 8 intelligence
    """
    def __init__(self):
        super(PFC, self).__init__()
        self.NeoCortex = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.DLPFC = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )
    
    def forward(self, x):
        """
        x: input for Prefrontal Cortex (characteristics2)
        """
        out = self.NeoCortex(x)
        out = self.DLPFC(out)
        return out