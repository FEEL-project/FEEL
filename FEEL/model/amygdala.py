import torch
import torch.nn as nn
from .subcortical_pathway import SubcorticalPathway
from .controller import EvalController

class Amygdala():
    """扁桃体: 感情評価を行う
    SubcorticalPath: x1 (rough characteristics) -> evaluation1 (impulsive emotion)
    EvalController: evaluation1, pre_evaluation -> evaluation2 (deliberate emotion)
    """
    def __init__(self, SubcorticalPathway: SubcorticalPathway, EvalController: EvalController):
        super(Amygdala, self).__init__()
        self.meditation = False     # if True, the model is in meditation mode
        self.SubcorticalPath = SubcorticalPathway
        self.evaluation1 = None     # evaluation by Subcortical Pathway
        self.EvalController = EvalController
        self.evaluation2 = None     # evaluation by Evaluation Controller
        
    def forward(self, x1=None, evaluation1=None, pre_evaluation=None):
        """
        x1: input for Subcortical Pathway (characteristics1)
        x2: input for Evaluation Controller (pre_evaluation)
        """
        if self.meditation:
            self.evaluation1 = evaluation1
        else:
            self.evaluation1 = self.SubcorticalPath(x1)
        out = self.EvalController(self.evaluation1, pre_evaluation)
        self.evaluation2 = out
        return out


# 以下、別ファイルに移管予定
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = dataloader.shape[0]
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, threshold=0.1):
    model.eval()  # モデルを評価モードに設定
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, within_threshold = 0, 0

    with torch.no_grad():  # 勾配計算を無効化
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()  # yの次元を合わせる
            # 予測値とターゲット値の絶対誤差が閾値以下である割合
            within_threshold += (torch.abs(pred - y.unsqueeze(1)) < threshold).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = within_threshold / size  # 閾値内に収まる予測の割合

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")



loss_fn = nn.MSELoss()