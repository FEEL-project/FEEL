import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib2 import Path
import json
import csv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoDataset(Dataset):
    def __init__(self, inputs, labels, names,clip_length, frame_size=(224, 224)):
        # 動画ファイルパスのリスト
        # 正解ラベルのリストをテンソルに変換
        self.inputs = inputs
        self.labels = labels
        # サンプリングするフレームクリップ数
        self.clip_length = clip_length
        # フレームのリサイズサイズ
        self.frame_size = frame_size
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        label = self.labels[index]
        name = self.names[index]

        return inputs, label, name

    def save_to_file(self, file_path):
        """
        データセットをファイルに保存する関数。
        
        :param file_path: 保存先のファイルパス
        """
        # 保存用のデータ構造を作成
        data = {
            "inputs": [input.tolist() for input in self.inputs],
            "labels": [label.tolist() for label in self.labels],
            "names": self.names
        }
        # JSON ファイルとして保存
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

def csv_to_dict(file_path):
    result_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # ヘッダーをスキップ
        next(reader)
        for row in reader:
            key = row[0]
            values = row[1:]
            values = list(map(float, row[1:])) 
            values = torch.tensor(values).to(DEVICE)
            result_dict[key] = values
    return result_dict

def load_video_dataset(video_dir: str, label_path: str, batch_size: int, clip_length: int, mvit)->DataLoader:
# 動画データセットのディレクトリ
    video_dir_path = Path(video_dir)
    label_df = csv_to_dict(label_path)
    frame_size=(224, 224)
    # 動画ファイルパスとラベルの準備
    inputs = []
    labels = []
    names = []
    for video_file in video_dir_path.glob('*.mp4'):
        path = os.path.join(video_dir, video_file.name)
        # OpenCVで動画ファイルを開く
        cap = cv2.VideoCapture(path)
        # 動画の全フレーム数を取得
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 動画の全フレーム数からclip_lengthの数だけ均等なインデックスを取得
        indices = np.linspace(1, frame_count, num=clip_length, dtype=int)

        input = []
        count = 1
        while True:
            # フレームを順番に取得
            ret, frame = cap.read()
            # 正しく取得できればretにTrueが返される
            if ret:
                # 均等に取得したindicesリスト内のインデックスのときだけフレームを保存
                if count in indices:
                    # フレームをリサイズ
                    frame = cv2.resize(frame, frame_size)
                    input.append(frame)
            else:
                break
            count += 1

        cap.release()

        # 取得したフレームのリストをテンソルに変換し、[T, C, H, W] に整形
        input = np.array(input)  # [T, H, W, C]
        input = np.transpose(input, (3, 0, 1, 2))  # [C, T, H, W]

        input = torch.tensor(input, dtype=torch.float32) / 255.0  # 正規化
        input = input.unsqueeze(0)
        input = input.to(DEVICE)
        with torch.no_grad():
            _,input,_ = mvit(input)
        input = input.squeeze(0)
        print(input.shape)
        inputs.append(input)
        labels.append(label_df[video_file.name])
        names.append(video_file.name)

    # データセットとDataLoaderの作成
    data_set = VideoDataset(inputs, labels, names, clip_length=clip_length)
    data_set.save_to_file("/home/u01230/SoccerNarration/FEEL/dataset/video_dataset.json")
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


# python load_video_dataset('/home/ghoti/FEEL/FEEL/data/small_data/renamed', '/home/ghoti/FEEL/FEEL/annotation/params_test.csv', 2, 16)