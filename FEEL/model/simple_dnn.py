import torch
import torch.nn as nn


class char2eval(nn.Module):
    def __init__(self):
        super(char2eval, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(512, 8),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.linear_relu_stack(input)
        return output