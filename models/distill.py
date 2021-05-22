from .base import Base

class Distill(Base):
    def __init__(self, teacher, student, teacher_layer_list, student_layer_list, connectors, is_kd=False):
        self.teacher = teacher
        self.student = student
        self.teacher_layer_list = teacher_layer_list
        self.student_layer_list = student_layer_list
        self.connectors = connectors
        self.is_kd = is_kd


