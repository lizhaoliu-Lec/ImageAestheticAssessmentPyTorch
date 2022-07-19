"""
NIMA: Neural Image Assessment. TIP 2018

paper reference: https://arxiv.org/pdf/1709.05424
"""
import torch
import torch.nn as nn

from model.base import get_base
from model.factory import ModelFactory


@ModelFactory.register('NIMA')
class NIMA(nn.Module):
    def __init__(self,
                 base_name='vgg16',
                 num_classes=10,
                 drop_rate=0.75,
                 **kwargs):
        super().__init__()
        self.base_name = base_name
        self.base = get_base(base_name, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features=self.base.last_dim, out_features=num_classes),
            nn.Softmax(dim=-1))

    def forward(self, x):
        if 'resnet' in self.base_name:
            x = self.base(x, pool=True)
        else:
            x = self.base(x)
        return self.head(x)


if __name__ == '__main__':
    def run_nima():
        x = torch.randn((5, 3, 224, 224)).cuda()
        model = NIMA().cuda()
        out = model(x)
        print("===> x.size() ", x.size())
        print("===> out.size() ", out.size())


    run_nima()
