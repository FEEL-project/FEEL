import torch
import torch.nn as nn
from model import EnhancedMViT, Hippocampus, Amygdala, PrefrontalCortex

class Elaborator(nn.Module):
    def __init__(self):
        super(Elaborator, self).__init__()
        self.sensory_cortex = EnhancedMViT()
        self.hippocampus = Hippocampus()
        self.amygdala = Amygdala()
        self.prefrontal_cortex = PrefrontalCortex()
        
    def __meditation__(self):
        self.amygdala.meditation = True
        episode = self.hippocampus.generate_episode()

    def forward(self, x):
        x = self.hippocampus(x)
        x = self.amygdala(x)
        x = self.prefrontal_cortex(x)
        return x