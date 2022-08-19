import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152

from common import PROJECT_NAME
from model.utils import freeze_bn, name_size_grad, freeze_weights

import logging


class ResNetBase(nn.Module):
    NAME_2_RESNET = {
        _resnet.__name__: _resnet
        for _resnet in [resnet18, resnet34, resnet50, resnet101, resnet152]
    }

    def __init__(self, resnet_name='resnet50', pretrained=True, bn_frozen=False, **kwargs):
        logger = logging.getLogger(PROJECT_NAME)
        super().__init__()

        d2_path = None
        if 'load_from_d2' in kwargs:
            d2_path = kwargs.pop('load_from_d2')

        self.freeze_list = []
        if 'freeze_list' in kwargs:
            self.freeze_list = kwargs['freeze_list']
            kwargs.pop('freeze_list')
            for _ in self.freeze_list:
                assert _ in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

        if pretrained:
            logging.info("Using ImageNet Pretrained Parameters for model training")
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

        if d2_path is not None:
            from common.d2_convert import RESNET50_CONVERT_MAP

            os.path.exists(d2_path), '{} not exists'.format(d2_path)
            logging.info("Loading pretrained weight from d2")
            assert resnet_name == 'resnet50', 'convert map is only available for resnet50'
            d2_model = torch.load(d2_path)['model']
            converted_d2_model = {}
            for k, v in d2_model.items():
                if k in RESNET50_CONVERT_MAP:
                    converted_d2_model[RESNET50_CONVERT_MAP[k]] = v

            self.load_state_dict(converted_d2_model)
            logger.info("d2 model is successfully loaded!")

        if bn_frozen:
            logger.info("The BN in ResNetBase is frozen")
            freeze_bn(self)

        for freeze_name in self.freeze_list:
            logger.info("Freezing resnet's: {}".format(freeze_name))
            freeze_weights(self.__getattr__(freeze_name))

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
