import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import ReLU
from torchsummary import summary

from models import ConvBatchNormRelu, ConvBatchNormRelu6
from models import GetIntermediateLayer
from models import Distill


if __name__ == '__main__':
    teacher_model = nn.Sequential(
        ConvBatchNormRelu6(3, 4, kernel_size=3, padding=1),
        ConvBatchNormRelu6(4, 4, kernel_size=3, padding=1)
        )
    teacher_model.eval()
    student_model = nn.Sequential(
        ConvBatchNormRelu6(3, 4, kernel_size=3, padding=1),
        ConvBatchNormRelu6(4, 4, kernel_size=3, padding=1)
        )
    student_model.eval()
    x = torch.Tensor(2,3,5,5)
    distill_model = Distill(teacher_model, student_model, ['0.cbr.conv', '1.cbr.conv'], ['0.cbr.conv', '1.cbr.conv'], [nn.MSELoss(), nn.MSELoss()])
    output = distill_model(x)
    print(output)

    