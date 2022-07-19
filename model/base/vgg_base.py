import torch
import torch.nn as nn

from torchvision.models import vgg11
from torchvision.models import vgg13
from torchvision.models import vgg16
from torchvision.models import vgg19
from torchvision.models import vgg11_bn
from torchvision.models import vgg13_bn
from torchvision.models import vgg16_bn
from torchvision.models import vgg19_bn

from model.utils import freeze_bn, name_size_grad

import logging


class VGGBase(nn.Module):
    NAME_2_VGG = {
        _vgg.__name__: _vgg
        for _vgg in [vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn]
    }

    def __init__(self, vgg_name='vgg16', pretrained=True, bn_frozen=False, **kwargs):
        super().__init__()
        self.features = self.NAME_2_VGG[vgg_name](pretrained=pretrained, **kwargs).features

        self.last_dim = 512 * 7 * 7

        if bn_frozen and '_bn' in vgg_name:
            logging.info("The BN in VGGBase is frozen")
            freeze_bn(self)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        return out


if __name__ == '__main__':
    def run_VGGBase():
        model = VGGBase()
        x = torch.randn((5, 3, 224, 224))
        out = model(x)
        print("====> out.size() ", out.size())


    run_VGGBase()
