import torch
import torch.nn as nn
from model.utils import freeze_weights
from model.base import get_base
from model.factory import ModelFactory


@ModelFactory.register('ResNet')
class ResNet(nn.Module):
    def __init__(self,
                 base_name='resnet50',
                 stage='pretrained',
                 num_classes=2,
                 freeze_base=False,
                 **kwargs):
        super().__init__()
        assert stage in ['pretrained', 'finetune']
        self.base_name = base_name
        self.base = get_base(base_name, **kwargs)
        self.head = nn.Sequential(
            nn.Linear(in_features=self.base.last_dim, out_features=num_classes))
        self.stage = stage
        if freeze_base:
            freeze_weights(self.base)

    def forward(self, x):
        x = self.base(x, pool=True)
        return self.head(x)
