import torch
from torch import nn
from torchvision.models.video import mvit_v1_b, _unsqueeze

from SoccerNarration.FEEL.dataset.video_dataset import load_video_dataset


class my_MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = mvit_v1_b()

        # self.head = nn.Sequential(
        #     nn.Dropout(dropout, inplace=True),
        #     nn.Linear(block_setting[-1].output_channels, num_classes),
        # )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.head(x)

        return x
    

def default_mvit():
    video_dir = '/home/u01230/SoccerNarration/small_data'
    batch_size = 2
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)


    # モデルの準備と推論
    model = mvit_v1_b()
    model.eval()
    for inputs, labels in train_loader:
        print("Input shape:", inputs.shape)  # 入力テンソルの形状
        with torch.no_grad():
            outputs = model(inputs)
            print("Model output shape:", outputs.shape)  # 出力の形状
            print(outputs)


if __name__ == "__main__":
    default_mvit()