import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152

from model.utils import freeze_bn, name_size_grad

import logging


class ResNetBase(nn.Module):
    NAME_2_RESNET = {
        _resnet.__name__: _resnet
        for _resnet in [resnet18, resnet34, resnet50, resnet101, resnet152]
    }

    def __init__(self, resnet_name='resnet50', pretrained=True, bn_frozen=False, **kwargs):
        super().__init__()
        resnet = self.NAME_2_RESNET[resnet_name](pretrained=pretrained, **kwargs)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.last_dim = 512 if resnet_name in ['resnet18', 'resnet34'] else 2048
        if bn_frozen:
            logging.info("The BN in ResNetBase is frozen")
            freeze_bn(self)

    def forward(self, x, pool=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if pool:
            N, c = x.size(0), x.size(1)
            x = F.adaptive_avg_pool2d(x, (1, 1)).reshape((N, c))
        return x


if __name__ == '__main__':
    import torch


    def run_size():
        x = torch.randn((5, 3, 224, 224))
        model = ResNetBase(resnet_name='resnet34', pretrained=False)
        print("===> x.size() ", x.size())
        out = model(x)
        print("===> out.size() ", out.size())

        x = x.cuda()
        model.cuda()
        print("===> x.size() ", x.size())
        out = model(x)
        print("===> out.size() ", out.size())


    def run_freeze_bn():
        model = ResNetBase(resnet_name='resnet34', pretrained=False, bn_frozen=True)
        name_size_grad(model)


    run_freeze_bn()
