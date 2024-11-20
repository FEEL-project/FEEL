import torch
import torch.nn as nn
from .mvit import EnhancedMViT
from .amygdala import Amygdala
from .prefrontal_cortex import PFC
from .hippocampus import Hippocampus
from .controller import EvalController

class Elaborator(nn.Module):
    def __init__(self):
        super(Elaborator, self).__init__()
        self.sensory_cortex = EnhancedMViT()
        self.hippocampus = Hippocampus()
        self.amygdala = Amygdala()
        self.prefrontal_cortex = PFC()
        self.controller = EvalController()
    def __meditation__(self):
        self.amygdala.meditation = True
        episode = self.hippocampus.generate_episode()

    def forward(self, x):
        x = self.hippocampus(x)
        x = self.amygdala(x)
        x = self.prefrontal_cortex(x)
        return x

    #全体ぐるぐる
    #optimizerは二つのモデルに共通
    def train_loop(self, optimizer):
        self.prefrontal_cortex.train()
        self.controller.train()
        for i in range(self.hippocampus.num_events):
            event = self.hippocampus.get_event(i)
            eval_1 = event["eval_1"]
            episode = self.hippocampus.generate_episode(event)
            pre_eval = self.prefrontal_cortex(episode)
            amy_eval = self.amygdala(eval_1, pre_eval)
            loss = nn.MSELoss(amy_eval, event["eval_2"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_loop(self):
        self.prefrontal_cortex.eval()
        self.controller.eval()
        total_loss = 0
        for i in range(self.hippocampus.num_events):
            event = self.hippocampus.get_event(i)
            eval_1 = event["eval_1"]
            episode = self.hippocampus.generate_episode(event)
            with torch.no_grad():
                pre_eval = self.prefrontal_cortex(episode)
                amy_eval = self.amygdala(eval_1, pre_eval)
                loss = nn.MSELoss(amy_eval, event["eval_2"])
                total_loss += loss.item()
        print(f"Test Error: \n Avg loss: {total_loss/self.hippocampus.num_events:>8f} \n")
    
    def train2(self, optimizer):
        self.prefrontal_cortex.eval()
        self.controller.train()
        for i in range(self.hippocampus.num_events):
            event = self.hippocampus.get_event(i)
            eval_1 = event["eval_1"]
            episode = self.hippocampus.generate_episode(event)
            with torch.no_grad():
                pre_eval = self.prefrontal_cortex(episode)
            amy_eval = self.amygdala(eval_1, pre_eval)
            loss = nn.MSELoss(amy_eval, pre_eval)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test2(self):
        self.prefrontal_cortex.eval()
        self.controller.eval()
        total_loss = 0
        for i in range(self.hippocampus.num_events):
            event = self.hippocampus.__get_characteristics__(i)
            eval_1 = event["eval_1"]
            episode = self.hippocampus.generate_episode(event)
            with torch.no_grad():
                pre_eval = self.prefrontal_cortex(episode)
                amy_eval = self.amygdala(eval_1, pre_eval)
                loss = nn.MSELoss(amy_eval, pre_eval)
                total_loss += loss.item()
        print(f"Test Error: \n Avg loss: {total_loss/self.hippocampus.num_events:>8f} \n")
