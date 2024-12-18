import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib2 import Path
import json
import csv
from functools import wraps
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DATASET_CACHE = True # Whether to load from cache instead of processing video every time
VIDEO_DATASET_PATH = "dataset/splitted_TrainVal.json" # Path to save/load dataset cache

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

    @classmethod
    def load_from_file(cls, file_path: str, clip_length: int) -> "VideoDataset":
        """Load VideoDataset from file

        Args:
            file_path (str): File path to load
            clip_length (int): Clip lenth

        Returns:
            VideoDataset: VideoDataset instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        inputs = [torch.Tensor(input).to(DEVICE) for input in data.get("inputs")]
        labels = [torch.Tensor(label).to(DEVICE) for label in data.get("labels")]
        names = data.get("names")
        self = cls(inputs=inputs, labels=labels, names=names, clip_length=clip_length)
        return self
        
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

def load_video_dataset(video_dir: str, label_path: str, batch_size: int, clip_length: int, mvit, use_cache: bool = True, cache_path: str = None)->DataLoader:
# 動画データセットのディレクトリ
    data_set: VideoDataset = None
    if use_cache and cache_path is not None and os.path.exists(cache_path):
        logging.info(f"Loading dataset from file {cache_path}")
        data_set = VideoDataset.load_from_file(cache_path, clip_length)
    else:
        logging.info(f"Processing video files in {video_dir}")
        video_dir_path = Path(video_dir)
        label_df = csv_to_dict(label_path)
        frame_size=(224, 224)
        # 動画ファイルパスとラベルの準備
        inputs = []
        labels = []
        names = []
        for video_file in video_dir_path.glob('*.mp4'):
            logging.info(f"Processing {video_file.name}")
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
            inputs.append(input)
            labels.append(label_df[video_file.name])
            names.append(video_file.name)

        # データセットとDataLoaderの作成
        data_set = VideoDataset(inputs, labels, names, clip_length=clip_length)
        if cache_path is not None:
            data_set.save_to_file(cache_path)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)
    

# python load_video_dataset('/home/ghoti/FEEL/FEEL/data/small_data/renamed', '/home/ghoti/FEEL/FEEL/annotation/params_test.csv', 2, 16)

def load_video(video_path: str, clip_length: int, mvit, frame_size=(224, 224)):
    """
    特定の動画の特徴量を取得する関数。
    
    Args:
        video_path (str): 特徴量を取得したい動画のパス
        clip_length (int): 抽出するフレームクリップ数
        mvit: 動画特徴量抽出モデル
        frame_size (tuple): フレームのリサイズサイズ (デフォルト: (224, 224))
    
    Returns:
        torch.Tensor: 動画の特徴量テンソル
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    # OpenCVで動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    # 動画の全フレーム数を取得
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 動画の全フレーム数からclip_lengthの数だけ均等なインデックスを取得
    indices = np.linspace(1, frame_count, num=clip_length, dtype=int)

    input_frames = []
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
                input_frames.append(frame)
        else:
            break
        count += 1

    cap.release()

    # 取得したフレームのリストをテンソルに変換し、[T, C, H, W] に整形
    input_frames = np.array(input_frames)  # [T, H, W, C]
    input_frames = np.transpose(input_frames, (3, 0, 1, 2))  # [C, T, H, W]

    input_tensor = torch.tensor(input_frames, dtype=torch.float32) / 255.0  # 正規化
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # 動画特徴量をモデルで抽出
    with torch.no_grad():
        _, features, _ = mvit(input_tensor)

    print(features.shape)
    return features
