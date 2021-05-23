import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import ReLU
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torch.nn import init

from models import ConvBatchNormRelu, ConvBatchNormRelu6
from models import GetIntermediateLayer
from models import Distill
from config import Config
from data.cifar100 import train_dataset, val_dataset

config = Config()


if __name__ == '__main__':

    teacher_model = mobilenet_v3_large(pretrained=True)
    student_model = mobilenet_v3_large(pretrained=True)
    torch.nn.init.xavier_uniform_(student_model.features[2].block[2][0].weight)
    torch.nn.init.xavier_uniform_(student_model.features[3].block[2][0].weight)
    print(torch.allclose(teacher_model.features[2].block[1][0].weight, student_model.features[2].block[1][0].weight))

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                              pin_memory=True, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size,
                            pin_memory=True, shuffle=False, num_workers=config.num_workers)
    teacher_layers = [
        # 'features.0.0',
        # 'features.1.block.0.0',
        # 'features.1.block.1.0',
        # 'features.2.block.0.0',
        # 'features.2.block.1.0',
        'features.2.block.2.0',
        # 'features.3.block.0.0',
        # 'features.3.block.1.0',
        'features.3.block.2.0',
    ]

    student_layers = [
        # 'features.0.0',
        # 'features.1.block.0.0',
        # 'features.1.block.1.0',
        # 'features.2.block.0.0',
        # 'features.2.block.1.0',
        'features.2.block.2.0',
        # 'features.3.block.0.0',
        # 'features.3.block.1.0',
        'features.3.block.2.0',
    ]

    connectors = [nn.MSELoss() for _ in range(len(teacher_layers))]
    distill_model = Distill(teacher_model, student_model, teacher_layers, student_layers, connectors, config)
    trainer = pl.Trainer(
        weights_summary=None,
        gpus=1,
        # max_epochs=1,
        # limit_train_batches=0.1
    )
    trainer.fit(distill_model, train_loader, val_loader)
    print(torch.allclose(teacher_model.features[2].block[1][0].weight, student_model.features[2].block[1][0].weight))
