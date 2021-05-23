from functools import partial

from .base import Base

class GetIntermediateLayer(Base):
    def __init__(self):
        super().__init__()
        self.profile = {}
    
    def hook_fn(self, module, name):
        def hook(m ,i, o):
            self.profile[name] = {}
            self.profile[name]['module'] = m
            self.profile[name]['input'] = i[0]
            self.profile[name]['output'] = o
        module.register_forward_hook(hook)

    def register_forward_hook(self, model, layer_list):
        for name, module in model.named_modules():
            if name in layer_list:
                self.hook_fn(module, name)

