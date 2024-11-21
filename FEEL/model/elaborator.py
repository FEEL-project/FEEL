import pickle
import torch
import torch.nn as nn
from .mvit import EnhancedMViT
from .amygdala import Amygdala
from .prefrontal_cortex import PFC
from .hippocampus import Hippocampus
from .subcortical_pathway import SubcorticalPathway
from .controller import EvalController
from .subcortical_pathway import SubcorticalPathway

class Elaborator(nn.Module):
    def __init__(self, 
                 pretrained_sensory_cortex: str = None, # path to pretrained sensory cortex (pth)
                 pretrained_hippocampus: str = None,    # path to hippocampus state (pickle)
                 pretrained_subcortical_pathway: str = None,       # path to pretrained amygdala (pth)
                 pretrained_controller: str = None,  # path to pretrained controller (pth)
                 pretrained_prefrontal_cortex: str = None,  # path to pretrained prefrontal cortex (pth)
                 ):
        super(Elaborator, self).__init__()
        self.sensory_cortex = EnhancedMViT()
        if pretrained_sensory_cortex is not None:
            self.sensory_cortex.load_state_dict(torch.load(pretrained_sensory_cortex))
        if pretrained_hippocampus is not None:
            self.hippocampus = Hippocampus.load_from_file(pretrained_hippocampus)
        else:
            self.hippocampus = Hippocampus()
        self.prefrontal_cortex = PFC()
        if pretrained_sensory_cortex is not None:
            self.sensory_cortex.load_state_dict(torch.load(pretrained_sensory_cortex))
        self.subcortical_pathway = SubcorticalPathway()
        if pretrained_subcortical_pathway is not None:
            self.subcortical_pathway.load_state_dict(torch.load(pretrained_subcortical_pathway))
        self.controller = EvalController()
        if pretrained_controller is not None:
            self.controller.load_state_dict(torch.load(pretrained_controller))
        self.amygdala = Amygdala(self.subcortical_pathway, self.controller)
        
    def __meditation__(self):
        self.amygdala.meditation = True
        episode = self.hippocampus.generate_episode()

    def forward(self, x):
        """Simpliest forward pass
        1. video is the only input
        2. not save event to hippocampus
        """
        x = self.sensory_cortex(x)
        event = self.hippocampus.receive(x)
        episode = self.hippocampus.generate_episode(event)
        pre_eval = self.prefrontal_cortex(episode)
        evaluation = self.amygdala(x1 = x, pre_evaluation = pre_eval)
        return evaluation

    # train loop, test loop
    # optimizerは二つのモデルに共通
    def train_maximization(self, optimizer):
        """
        bring amy_eval closer to eval_2 (annotated amygdala evaluation)
        in order to train the prefrontal cortex
        """
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
        for i in range(self.hippocampus.num_events): #FIXME: Probably fails when there is a deleted memory
            event = self.hippocampus.get_event(i)
            eval_1 = event["eval_1"]
            episode = self.hippocampus.generate_episode(event)
            with torch.no_grad():
                pre_eval = self.prefrontal_cortex(episode)
                amy_eval = self.amygdala(eval_1, pre_eval)
                loss = nn.MSELoss(amy_eval, event["eval_2"])
                total_loss += loss.item()
        print(f"Test Error: \n Avg loss: {total_loss/self.hippocampus.num_events:>8f} \n")
    
    def train_expectation(self, optimizer):
        """
        bring amy_eval closer to pre_eval
        in order to train the controller and make PFC more dominant
        """
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
            event = self.hippocampus.get_event(i)
            eval_1 = event["eval_1"]
            episode = self.hippocampus.generate_episode(event)
            with torch.no_grad():
                pre_eval = self.prefrontal_cortex(episode)
                amy_eval = self.amygdala(eval_1, pre_eval)
                loss = nn.MSELoss(amy_eval, pre_eval)
                total_loss += loss.item()
        print(f"Test Error: \n Avg loss: {total_loss/self.hippocampus.num_events:>8f} \n")