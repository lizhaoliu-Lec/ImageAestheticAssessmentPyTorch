import torch.nn as nn

from loss.factory import LossFactory


@LossFactory.register('CrossEntropyLoss')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    ...


@LossFactory.register('MSELoss')
class MSELoss(nn.MSELoss):
    ...
