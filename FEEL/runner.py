import torch
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

from dataset.video_dataset import load_video_dataset
from model import EnhancedMViT, Elaborator

def default_mvit(video_dir: str):
    batch_size = 1
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)


    # モデルの準備と推論
    model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
    model.eval()
    for inputs, labels in train_loader:
        with torch.no_grad():
            outputs = model(inputs)
            # print("Model output shape:", outputs.shape)  # 出力の形状
            max_index = torch.argmax(outputs)
            print(max_index)
            print(labels)

def enhanced_mvit(video_dir: str):
    batch_size = 1
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)


    # モデルの準備と推論
    model = EnhancedMViT(pretrained=True)
    model.eval()
    for inputs, labels in train_loader:
        with torch.no_grad():
            c1,c2,outputs = model(inputs)
            # print("Model output shape:", outputs.shape)  # 出力の形状
            max_index = torch.argmax(outputs)
            print("c1")
            print(c1.shape)
            print("c2")
            print(c2.shape)
            # print(max_index)
            # print(labels)
            break

def seeing(video_dir: str):
    """
    動画を渡して、評価を出力させる
    """
    batch_size = 1
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)
    elaborator = Elaborator()
    mvit = EnhancedMViT(pretrained=True)
    elaborator.eval()
    mvit.eval()
    for inputs, labels in train_loader:
        with torch.no_grad():
            c1, c2, outputs = mvit(inputs)
            # event = elaborator.hippocampus.receive(characteristics=c2, evaluation=labels, event_id=0)
            # episode = elaborator.hippocampus.generate_episode(event)
            pre_eval = elaborator.prefrontal_cortex(c2)
            # pre_eval = elaborator.prefrontal_cortex(episode)
            eval_1 = elaborator.subcortical_pathway(c2)
            eval_2 = elaborator.controller(eval_1, pre_eval)
            print("pre_eval")
            print(pre_eval)
            print("eval_1")
            print(eval_1)
            print("eval_2")
            print(eval_2)
            break


if __name__ == "__main__":
    seeing("/home/ghoti/FEEL/FEEL/data/small_data/renamed")