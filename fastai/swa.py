import torch
from .sgdr import *
from .core import *


class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model,self.swa_model,self.swa_start=model,swa_model,swa_start
        
    def on_train_begin(self):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, metrics):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1
            
        self.epoch += 1
            
    def update_average_model(self):
        # update running average of parameters
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)            
    
def collect_bn_modules(module, bn_modules):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        bn_modules.append(module)

def fix_batchnorm(swa_model, train_dl):
    bn_modules = []
    swa_model.apply(lambda module: collect_bn_modules(module, bn_modules))
    
    if not bn_modules:
        return

    swa_model.train()

    for module in bn_modules:
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)
    
    momenta = [m.momentum for m in bn_modules]

    inputs_seen = 0

    for (*x,y) in iter(train_dl):        
        xs = V(x)
        batch_size = xs[0].size(0)

        momentum = batch_size / (inputs_seen + batch_size)
        for module in bn_modules:
            module.momentum = momentum
                            
        res = swa_model(*xs)        
        
        inputs_seen += batch_size
                
    for module, momentum in zip(bn_modules, momenta):
        module.momentum = momentum    