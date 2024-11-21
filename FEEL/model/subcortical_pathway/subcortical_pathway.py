import torch
import torch.nn as nn

class SubcorticalPathway(nn.Module):
    """
    扁桃体: 粗い情報をもとに、速い経路(Subcortical Pathway)で threat-neutral-reward の判断を行う 
    """
    def __init__(self):
        super(SubcorticalPathway, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()   # [-1, 1]の範囲に正規化
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        