from .base import Base
from functools import partial

class GetIntermediateLayer(Base):
    def __init__(self):
        super().__init__()
        self.output = {}
    
    def hook_fn(self, module, name):
        def hook(m ,i, o):
            self.output[name] = o
        module.register_forward_hook(hook)

    def register_forward_hook(self, model, layer_list):
        for name, module in model.named_modules():
            if name in layer_list:
                self.hook_fn(module, name)

