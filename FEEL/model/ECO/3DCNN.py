"""
ECOの3DNetモジュールである3DCNNを実装する
モデルのユニットテスト方法:
    net = ECO_3D()
    net.train()
"""
# インポート
import torch
from torch import nn

class Resnet_3D_3(nn.Module):
    """Resnet_3D_3
        (ch=96,frames=16,height=28,width=28)
        ->Conv3d->+(
            B+R->Conv3d(kernel=3)+B+R->Conv3d(3),
            Identity
        )->B+R->(ch=128,16,28,28)
    """

    def __init__(self):
        super(Resnet_3D_3, self).__init__()
        
        self.res3a_2 = nn.Conv3d(96, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res3a_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res3b_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.res3a_2(x)
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_relu(out)
        out = self.res3b_2(out)

        out += residual

        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out

class Resnet_3D_4(nn.Module):
    """Resnet_3D_4
        (ch=128,frames=16,height=28,width=28)
        ->+(
            Conv3d(kernel=3)+B+R->Conv3d(3),
            Identity
        )->+(
            B+R->Conv3d(kernel=3)+B+R->Conv3d(3),
            Identity
        )->B+R->(ch=256,8,14,14)
    """

    def __init__(self):
        super(Resnet_3D_4, self).__init__()

        self.res4a_1 = nn.Conv3d(128, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.res4a_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_1_relu = nn.ReLU(inplace=True)
        self.res4a_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res4a_down = nn.Conv3d(128, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.res4a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_relu = nn.ReLU(inplace=True)
        
        self.res4b_1 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res4b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_1_relu = nn.ReLU(inplace=True)
        self.res4b_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res4b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res4a_down(x)

        out = self.res4a_1(x)
        out = self.res4a_1_bn(out)
        out = self.res4a_1_relu(out)

        out = self.res4a_2(out)

        out += residual

        residual2 = out

        out = self.res4a_bn(out)
        out = self.res4a_relu(out)

        out = self.res4b_1(out)

        out = self.res4b_1_bn(out)
        out = self.res4b_1_relu(out)

        out = self.res4b_2(out)

        out += residual2

        out = self.res4b_bn(out)
        out = self.res4b_relu(out)

        return out

class Resnet_3D_5(nn.Module):
    """Resnet_3D_5
        (ch=256,frames=8,height=28,width=28)
        ->+(
            Conv3d(kernel=3)+B+R->Conv3d(3),
            Conv3d(3)
        )->+(
            B+R->Conv3d(kernel=3)+B+R->Conv3d(3),
            Identity
        )->B+R->(ch=512,4,7,7)
    """

    def __init__(self):
        super(Resnet_3D_5, self).__init__()
        
        self.res5a_1 = nn.Conv3d(256, 512, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.res5a_1_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_1_relu = nn.ReLU(inplace=True)
        self.res5a_2 = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res5a_down = nn.Conv3d(256, 512, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.res5a_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_relu = nn.ReLU(inplace=True)
        
        self.res5b_1 = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res5b_1_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_1_relu = nn.ReLU(inplace=True)
        self.res5b_2 = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res5b_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res5a_down(x)

        out = self.res5a_1(x)
        out = self.res5a_1_bn(out)
        out = self.res5a_1_relu(out)

        out = self.res5a_2(out)

        out += residual  # res5a

        residual2 = out

        out = self.res5a_bn(out)
        out = self.res5a_relu(out)

        out = self.res5b_1(out)

        out = self.res5b_1_bn(out)
        out = self.res5b_1_relu(out)

        out = self.res5b_2(out)

        out += residual2  # res5b

        out = self.res5b_bn(out)
        out = self.res5b_relu(out)

        return out

class ECO_3D(nn.Module):
    """ECO3D
        (batchsize,frames=16,ch=96,28,28)->(batchsize,ch=96,frames=16,28,28)
        ->InceptionA->InceptionB->InceptionC->(batchsize,ch=512,1,1,1)->(batchsize,ch=512)
    """
    def __init__(self):
        super(ECO_3D, self).__init__()

        # 3D_Resnetジュール
        self.res_3d_3 = Resnet_3D_3()
        self.res_3d_4 = Resnet_3D_4()
        self.res_3d_5 = Resnet_3D_5()

        # Global Average Pooling
        self.global_pool = nn.AvgPool3d(
            kernel_size=(4, 7, 7), stride=1, padding=0)

    def forward(self, x):
        '''
        入力xのサイズtorch.Size([batch_num,frames, 96, 28, 28]))
        '''
        out = torch.transpose(x, 1, 2)  # テンソルの順番入れ替え
        # out = out.permute(0,2,1,3,4) と同じこと
        out = self.res_3d_3(out)
        out = self.res_3d_4(out)
        out = self.res_3d_5(out)
        out = self.global_pool(out)
        
        # テンソルサイズを変更
        # torch.Size([batch_num, 512, 1, 1, 1])からtorch.Size([batch_num, 512])へ
        out =out.view(out.size()[0], out.size()[1])
        
        return out