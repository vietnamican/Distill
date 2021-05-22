from .base import Base

class GetIntermediateLayer(Base):
    def __init__(self, model, layer_list):
        self.model = model
        self.layer_list = layer_list
    
    # def forward(self, x):

