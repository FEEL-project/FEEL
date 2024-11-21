import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True) #FIXME: Remove this line

class SubcorticalPathway(nn.Module):
    """
    扁桃体: 粗い情報をもとに、速い経路(Subcortical Pathway)で threat-neutral-reward の判断を行う 
    """
    def __init__(self):
        super(SubcorticalPathway, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 1),
            nn.Tanh()   # [-1, 1]の範囲に正規化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        