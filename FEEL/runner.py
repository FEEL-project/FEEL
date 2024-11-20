import torch
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

from dataset.video_dataset import load_video_dataset
from model import EnhancedMViT, PFC

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
    model_pfc = PFC(768, 2, 8).to(device=torch.device("cpu"))
    model.eval()
    model_pfc.eval()
    for inputs, labels in train_loader:
        with torch.no_grad():
            c1,c2,outputs = model(inputs)
            c1_,c2_,outputs_ = model(inputs)
            cat = torch.cat((c2, c2_)).unsqueeze(1)
            print("cat", cat.shape)
            pfc_output = model_pfc(cat)
            print(pfc_output)
            print(pfc_output.shape)
            break

if __name__ == "__main__":
    enhanced_mvit("data/small_data/renamed")