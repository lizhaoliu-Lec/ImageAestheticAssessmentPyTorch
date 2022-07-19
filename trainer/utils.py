import torch
import torch.nn as nn


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return super().__getattr__('module')
        else:
            return getattr(self.module, name)


def get_gpu_device(gpu_id: str):
    return torch.device("cuda:%s" % gpu_id)
