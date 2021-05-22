import torch
import pytorch_lightning as pl
from torch.nn import ReLU
from torchsummary import summary

from models import ConvBatchNormRelu, ConvBatchNormRelu6

if __name__ == '__main__':
    model = ConvBatchNormRelu6(3, 64, kernel_size=3, padding=0)
    model.eval()
    # model = model.to('cpu')
    x = torch.Tensor(2,3,32,32)
    summary(model, x, device='cpu')
    print(model)
    