import os

import torch
import torch.nn as nn

def show_inference_result(dir_path, outputs_input, id_label_dict, idx=0):
    """ミニバッチの各データに対して、推論結果の上位を出力する関数を定義"""
    print("ファイル：", dir_path[idx])  # ファイル名

    outputs = outputs_input.clone()  # コピーを作成

    for i in range(5):
        """1位から5位までを表示"""
        output = outputs[idx]
        _, pred = torch.max(output, dim=0)  # 確率最大値のラベルを予測
        class_idx = int(pred.numpy())  # クラスIDを出力
        print("予測第{}位：{}".format(i+1, id_label_dict[class_idx]))
        outputs[idx][class_idx] = -1000  # 最大値だったものを消す（小さくする）
