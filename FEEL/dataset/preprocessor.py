"""
動画データを使い、ECO用のDataLoaderを作成する
train, testの前段階にて
    config = DatasetConfig(
        root_path='FEEL/data/',
        label_dict_path='FEEL/dataset/ActivityNet/Crawler/Kinetics/data/kinetics-400_train.csv',
    )
    val_dataloader = config.data_preprocess()
"""
import os
import argparse

import torch
import torch.nn as nn

from .preprocess import make_datapath_list, VideoTransform, get_label_id_dictionary, VideoDataset

# vieo_listの作成
# root_path = './data/kinetics_videos/'

class DatasetConfig():
    """
    Datasetの作成に必要なパラメータおよびDataset, DataLoaderのクラス
    """

    def __init__(self, root_path, label_dict_path,
                resize=224, crop_size=224, mean=[104,117,123], std=[1,1,1],
                num_segments=16, phase="val", batch_size=8,
                img_tmpl='image_{:05d}.jpg'):
        # path
        self.root_path = root_path  # 動画画像のフォルダへのパスリスト
        self.label_dicitionary_path = label_dict_path  # ラベルの辞書を保管するディレクトリ
        self.id_label_dict = None
        self.label_id_dict = None
        # 前処理パラメータ
        self.resize = resize
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        # Datasetクラスのパラメータ
        self.num_segments = num_segments  # 動画を何分割して使用するのかを決める
        self.phase = phase  # train or val
        self.transform = None  # 前処理
        self.img_tmpl = img_tmpl # 画像ファイル名の形式
        # データセット
        self.val_dataset = None     # 検証用
        self.train_dataset = None   # 訓練用
        self.test_dataset = None    # 推論用
        # torch.utils.data.DataLoaderクラスのパラメータ
        self.batch_size = batch_size    # バッチサイズ
        # データローダー
        self.train_dataloader = None    # 訓練用
        self.val_dataloader = None      # 検証用
        self.test_dataloader = None     # 推論用

    def data_preprocess(self, confirmation=False,):
        # vieo_listの作成
        video_list = make_datapath_list(self.root_path)

        # 前処理の設定
        video_transform = VideoTransform(self.resize, self.crop_size, self.mean, self.std)

        # ラベル辞書の作成
        # label_dicitionary_path = './video_download/kinetics_400_label_dicitionary.csv'
        self.label_id_dict, self.id_label_dict = get_label_id_dictionary(self.label_dicitionary_path)

        # Datasetの作成
        # num_segments は 動画を何分割して使用するのかを決める
        self.val_dataset = VideoDataset(video_list, self.label_id_dict, num_segments=self.num_segments,
                                phase=self.phase, transform=video_transform, img_tmpl=self.img_tmpl)

        # DataLoaderにします
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # 動作確認
        if confirmation:
            batch_iterator = iter(self.val_dataloader)  # イテレータに変換
            imgs_transformeds, labels, label_ids, dir_path = next(
                batch_iterator)  # 1番目の要素を取り出す
            print(imgs_transformeds.shape)
        return self.val_dataloader