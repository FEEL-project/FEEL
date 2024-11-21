import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib2 import Path
import csv

class VideoDataset(Dataset):
    def __init__(self, paths, labels, clip_length, frame_size=(224, 224)):
        # 動画ファイルパスのリスト
        self.paths = paths
        # 正解ラベルのリストをテンソルに変換
        self.labels = labels
        self.labels = torch.tensor(self.labels)
        # サンプリングするフレームクリップ数
        self.clip_length = clip_length
        # フレームのリサイズサイズ
        self.frame_size = frame_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]

        # OpenCVで動画ファイルを開く
        cap = cv2.VideoCapture(path)
        # 動画の全フレーム数を取得
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 動画の全フレーム数からclip_lengthの数だけ均等なインデックスを取得
        indices = np.linspace(1, frame_count, num=self.clip_length, dtype=int)

        inputs = []
        count = 1
        while True:
            # フレームを順番に取得
            ret, frame = cap.read()
            # 正しく取得できればretにTrueが返される
            if ret:
                # 均等に取得したindicesリスト内のインデックスのときだけフレームを保存
                if count in indices:
                    # フレームをリサイズ
                    frame = cv2.resize(frame, self.frame_size)
                    inputs.append(frame)
            else:
                break
            count += 1

        cap.release()

        # 取得したフレームのリストをテンソルに変換し、[T, C, H, W] に整形
        inputs = np.array(inputs)  # [T, H, W, C]
        inputs = np.transpose(inputs, (3, 0, 1, 2))  # [C, T, H, W]
        inputs = torch.tensor(inputs, dtype=torch.float32) / 255.0  # 正規化

        return inputs, label

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
            values = torch.tensor(values)
            result_dict[key] = values
    return result_dict

def load_video_dataset(video_dir: str, label_path: str, batch_size: int, clip_length: int)->DataLoader:
# 動画データセットのディレクトリ
    video_dir_path = Path(video_dir)
    label_df = csv_to_dict(label_path)
    # 動画ファイルパスとラベルの準備
    paths = []
    labels = []
    for video_file in video_dir_path.glob('*.mp4'):
        full_path = os.path.join(video_dir, video_file.name)
        paths.append(full_path)
        labels.append(label_df[video_file.name])

    # データセットとDataLoaderの作成
    data_set = VideoDataset(paths, labels, clip_length=clip_length)

    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

# python load_video_dataset('/home/ghoti/FEEL/FEEL/data/small_data/renamed', '/home/ghoti/FEEL/FEEL/annotation/params_test.csv', 2, 16)