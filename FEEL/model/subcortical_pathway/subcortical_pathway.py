import torch
import torch.nn as nn

class SubcorticalPathway(nn.Module):
    """
    扁桃体: 粗い情報をもとに、速い経路(Subcortical Pathway)で threat-neutral-reward の判断を行う 
    """
    def __init__(self):
        super(SubcorticalPathway, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # geluですが...
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
            nn.Tanh()   # [-1, 1]の範囲に正規化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        # return logits
        # activations = []
        # for layer in self.linear_gelu_stack:
        #     x = layer(x)
        #     activations.append(x)
        # return activations
        logits = self.linear_relu_stack(x)
        return logits