import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import ReLU
from torchsummary import summary
from torch.utils.data import DataLoader

from models import ConvBatchNormRelu, ConvBatchNormRelu6
from models import GetIntermediateLayer
from models import Distill
from config import Config
from data.cifar100 import train_dataset, val_dataset

config = Config()


if __name__ == '__main__':

    teacher_model = nn.Sequential(
        ConvBatchNormRelu6(3, 64, kernel_size=3, padding=1),
        ConvBatchNormRelu6(64, 64, kernel_size=3, padding=1)
        )
    student_model = nn.Sequential(
        ConvBatchNormRelu6(3, 64, kernel_size=3, padding=1),
        ConvBatchNormRelu6(64, 64, kernel_size=3, padding=1)
        )

    train_loader = DataLoader(train_dataset, batch_size = config.train_batch_size, pin_memory=True, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = config.val_batch_size, pin_memory=True, shuffle=False, num_workers=config.num_workers)
    
    distill_model = Distill(teacher_model, student_model, ['0.cbr.conv', '1.cbr.conv'], ['0.cbr.conv', '1.cbr.conv'], [nn.MSELoss(), nn.MSELoss()], config)
    trainer = pl.Trainer(
        weights_summary=None,
        gpus=1
    )
    trainer.fit(distill_model, train_loader, val_loader)