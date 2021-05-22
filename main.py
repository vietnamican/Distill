import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import ReLU
from torchsummary import summary

from models import ConvBatchNormRelu, ConvBatchNormRelu6
from models import GetIntermediateLayer

if __name__ == '__main__':
    model = nn.Sequential(
        ConvBatchNormRelu6(3, 64, kernel_size=3, padding=1),
        ConvBatchNormRelu6(64, 64, kernel_size=3, padding=1)
        )
    model.eval()
    x = torch.Tensor(2,3,32,32)
    summary(model, x, device='cpu')
    tracking_model = GetIntermediateLayer(model, ['0.cbr.conv', '1.cbr.conv'])
    print(tracking_model.output)
    tracking_model(x)
    print(tracking_model.output.keys())

    