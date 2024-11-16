import os
import argparse
import numpy as np

import torch
import torch.utils.data
from torch import nn

from dataset.preprocessor import DatasetConfig
from dataset.preprocess import VideoDataset
from model import ECO_Lite
from test.visualize import show_inference_result
from utils.load_pretrained import load_pretrained_ECO

def main():
    parser = argparse.ArgumentParser(description='Test: ECO')
    parser.add_argument('--root_path', '-r', default='./data/kinetics',
                       help='Path to ')
    # データローダーの作成
    config = DatasetConfig(
            root_path='./data/kinetics-400/val',
            label_dict_path='./dataset/label/kinetics-400_val.csv',
            # 前処理パラメータ
            resize=224, 
            crop_size=224, 
            mean=[104,117,123], 
            std=[1,1,1],
            # Datasetのパラメータ
            num_segments=16, 
            phase="val", 
            img_tmpl='image_{:05d}.jpg',
            # DataLoaderのパラメータ
            batch_size=8
        )
    val_dataloader = config.data_preprocess()

    # モデルインスタンスを生成
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPUが利用可能ならGPUを使用
    model = ECO_Lite().to(device)

    # 学習済みモデルをロード
    pretrained_path = './weights/'+'ECO_Lite_rgb_model_Kinetics.pth.tar'
    pretrained_model = torch.load(pretrained_path, map_location='cpu')
    pretrained_model_dict = pretrained_model['state_dict']

    # 自分が作成したモデルの変数名などを取得
    model_dict = model.state_dict()

    # 学習済みモデルから、自分が作成したモデルにロードするstate_dictを取得
    model_dict_new = load_pretrained_ECO(model_dict, pretrained_model_dict)

    # 自分が作成したモデルのパラメータを、学習済みのものに更新
    model.eval()
    model.load_state_dict(model_dict_new)

    # 推論
    model.eval()

    batch_iterator = iter(val_dataloader)
    imgs_transformeds, labels, label_ids, dir_path = next(
        batch_iterator)  # 1番目の要素を取り出す

    with torch.no_grad():
        outputs = model(imgs_transformeds)  # ECOで推論

    # print(outputs.shape)  # 出力のサイズ

    for idx in range(outputs.shape[0]):
        show_inference_result(dir_path, outputs, config.id_label_dict, idx)