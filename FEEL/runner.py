import torch
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

from dataset.video_dataset import load_video_dataset
from model import EnhancedMViT

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

if __name__ == "__main__":
    enhanced_mvit("/home/ghoti/FEEL/FEEL/data/small_data/renamed")