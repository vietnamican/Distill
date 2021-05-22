from .base import Base
from .get_intermediate_layer import GetIntermediateLayer

class Distill(Base):
    def __init__(self, teacher, student, teacher_layer_list, student_layer_list, connectors, is_kd=False):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_layer_list = teacher_layer_list
        self.student_layer_list = student_layer_list
        self.connectors = connectors
        self.is_kd = is_kd
        self.teacher_features_factory = GetIntermediateLayer()
        self.student_features_factory = GetIntermediateLayer()
        self.teacher_features_factory.register_forward_hook(teacher, teacher_layer_list)
        self.student_features_factory.register_forward_hook(student, student_layer_list)

    def forward(self, x):
        self.teacher(x)
        self.student(x)
        loss = 0
        teacher_outputs = self.teacher_features_factory.output
        student_outputs = self.student_features_factory.output
        for teacher_layer, student_layer, connector in zip(self.teacher_layer_list, self.student_layer_list, self.connectors):
            loss += connector(teacher_outputs[teacher_layer], student_outputs[student_layer])
        return loss
    


