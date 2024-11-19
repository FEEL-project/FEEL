import torch
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
from dataset.video_dataset import load_video_dataset

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

if __name__ == "__main__":
    default_mvit("/home/ghoti/FEEL/FEEL/data/small_data/renamed")