import torch
from torch import optim

from .base import Base
from .get_intermediate_layer import GetIntermediateLayer


class Distill(Base):
    def __init__(self, teacher, student, teacher_layer_list, student_layer_list, connectors, config, is_kd=False):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_layer_list = teacher_layer_list
        self.student_layer_list = student_layer_list
        self.connectors = connectors
        self.is_kd = is_kd
        self.teacher_features_factory = GetIntermediateLayer()
        self.student_features_factory = GetIntermediateLayer()
        self.teacher_features_factory.register_forward_hook(
            teacher, teacher_layer_list)
        self.student_features_factory.register_forward_hook(
            student, student_layer_list)
        self.config = config
        self.freeze_with_prefix('teacher')

    def forward(self, x):
        with torch.no_grad():
            self.teacher(x)
            self.student(x)
        return self.teacher_features_factory.profile, self.student_features_factory.profile

    def training_step(self, batch, batch_idx):
        x, y = batch
        teacher_profile, student_profile = self.forward(x)
        train_loss = 0
        for teacher_layer, student_layer, connector in zip(self.teacher_layer_list, self.student_layer_list, self.connectors):
            teacher_output = teacher_profile[teacher_layer]['module'](
                teacher_profile[teacher_layer]['input'])
            student_output = student_profile[student_layer]['module'](
                student_profile[student_layer]['input'])
            loss = connector(teacher_output, student_output)
            self.log("train_{}".format(teacher_layer), loss)
            train_loss += loss
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        teacher_profile, student_profile = self.forward(x)
        val_Loss = 0
        for teacher_layer, student_layer, connector in zip(self.teacher_layer_list, self.student_layer_list, self.connectors):
            teacher_output = teacher_profile[teacher_layer]['module'](
                teacher_profile[teacher_layer]['input'])
            student_output = student_profile[student_layer]['module'](
                student_profile[student_layer]['input'])
            loss = connector(teacher_output, student_output)
            self.log("val_{}".format(teacher_layer), loss)
            val_Loss += loss
        return val_Loss

    def configure_optimizers(self):
        max_epochs = self.config.max_epochs
        steps = [step*max_epochs for step in self.config.steps]
        optimizer = optim.SGD(self.student.parameters(),
                              lr=0.001, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, steps, gamma=0.1)
        return [optimizer], [lr_scheduler]
