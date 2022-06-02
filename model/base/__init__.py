from common import PROJECT_NAME
from .resnet_base import ResNetBase
from .vgg_base import VGGBase
from common.logging import logging


def get_base(base_name, **kwargs):
    logger = logging.getLogger(PROJECT_NAME)

    logger.info("Using base: {}".format(base_name))
    if 'base_name' in kwargs:
        kwargs.pop('base_name')
    if 'resnet' in base_name:
        return ResNetBase(resnet_name=base_name, **kwargs)
    if 'vgg' in base_name:
        return VGGBase(vgg_name=base_name, **kwargs)
