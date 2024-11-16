"""
Efficient Convolutional Network for Online Video Understanding Model
1. ECO_Lite: simple classifier
2. Full_ECO: bigger classifier
"""
import torch
import torch.nn as nn

from ECO import ECO_2D, ECO_3D

class ECO_Lite(nn.Module):
    """ECO_Lite"""
    def __init__(self, num_classes=400):
        super(ECO_Lite, self).__init__()

        self.eco_2d = ECO_2D()  # 2D Netモジュール
        self.eco_3d = ECO_3D()  # 3D Netモジュール
        self.fc_final = nn.Linear(in_features=512, out_features=num_classes, bias=True) # クラス分類の全結合層

    def forward(self, x):
        '''
        入力xはtorch.Size([batch_num, num_segments=16, 3, 224, 224]))
        '''

        # 入力xの各次元のサイズを取得する
        bs, ns, c, h, w = x.shape

        # xを(bs*ns, c, h, w)にサイズ変換する
        out = x.view(-1, c, h, w)
        """（注釈）
        PyTorchのConv2Dは入力のサイズが(batch_num, c, h, w)しか受け付けないため
        (batch_num, num_segments, c, h, w)は処理できない
        今は2次元画像を独立に処理するので、num_segmentsはbatch_numの次元に押し込んでも良いため
        (batch_num * num_segments, c, h, w)にサイズを変換する
        """

        # 2D Netモジュール 出力torch.Size([batch_num×16, 96, 28, 28])
        out = self.eco_2d(out)

        # 2次元画像をテンソルを3次元用に変換する
        # out = out.view(-1, ns, 96, 28, 28)  # num_segmentsをbatch_numの次元に押し込んだものを元に戻す
        out = out.view(-1, ns, out.size(1), out.size(2), out.size(3))  # num_segmentsをbatch_numの次元に押し込んだものを元に戻す

        # 3D Netモジュール 出力torch.Size([batch_num, 512])
        out = self.eco_3d(out)

        # クラス分類の全結合層　出力torch.Size([batch_num, class_num=400])
        out = self.fc_final(out)

        return out

class Full_ECO(nn.Module):
    """Full_ECO"""
    def __init__(self, num_classes=400):
        super(Full_ECO, self).__init__()

        self.eco_2d = ECO_2D()  # 2D Netモジュール
        self.eco_3d = nn.Sequential(
            ECO_3D(),  # 3D Netモジュール
            nn.Linear(in_features=512, out_features=num_classes, bias=True), # クラス分類の全結合層
            nn.Dropout(0.5)
        )
        # 追加の分類層
        self.classifier = nn.Sequential(
            nn.Linear(num_classes, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        '''
        入力xはtorch.Size([batch_num, num_segments=16, 3, 224, 224]))
        '''

        # 入力xの各次元のサイズを取得する
        bs, ns, c, h, w = x.shape

        # xを(bs*ns, c, h, w)にサイズ変換する
        out = x.view(-1, c, h, w)
        """（注釈）
        PyTorchのConv2Dは入力のサイズが(batch_num, c, h, w)しか受け付けないため
        (batch_num, num_segments, c, h, w)は処理できない
        今は2次元画像を独立に処理するので、num_segmentsはbatch_numの次元に押し込んでも良いため
        (batch_num * num_segments, c, h, w)にサイズを変換する
        """

        # 2D Netモジュール 出力torch.Size([batch_num×16, 96, 28, 28])
        out = self.eco_2d(out)

        # 2次元画像をテンソルを3次元用に変換する
        # out = out.view(-1, ns, 96, 28, 28)  # num_segmentsをbatch_numの次元に押し込んだものを元に戻す
        out = out.view(-1, ns, out.size(1), out.size(2), out.size(3))  # num_segmentsをbatch_numの次元に押し込んだものを元に戻す

        # 3D Netモジュール 出力torch.Size([batch_num, 512])
        out = self.eco_3d(out)

        # クラス分類の全結合層　出力torch.Size([batch_num, class_num=400])
        out = self.classifier(out)

        return out
    
