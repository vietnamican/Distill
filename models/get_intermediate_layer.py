from .base import Base
from functools import partial

class GetIntermediateLayer(Base):
    def __init__(self, model, layer_list):
        super().__init__()
        self.model = model
        self.layer_list = layer_list
        self.output = {}
        self.register_forward_hook()
    
    def hook_fn(self, module, name):
        def hook(m ,i, o):
            self.output[name] = o
        module.register_forward_hook(hook)

    def register_forward_hook(self):
        for name, module in self.model.named_modules():
            if name in self.layer_list:
                self.hook_fn(module, name)

    def forward(self, x):
        return self.model(x)


