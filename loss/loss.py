import torch.nn as nn
from torch import Tensor

from loss.factory import LossFactory


@LossFactory.register('CrossEntropyLoss')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    ...


@LossFactory.register('MSELoss')
class MSELoss(nn.MSELoss):
    ...


@LossFactory.register('MAELoss')
class MSELoss(nn.L1Loss):
    ...


@LossFactory.register('SmoothL1Loss')
class MSELoss(nn.SmoothL1Loss):
    ...


@LossFactory.register('CHAEDSmoothL1Loss')
class CHAEDSmoothL1Loss(nn.SmoothL1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input: (N, 3), target: (N)
        assert input.size()[1] == 3
        reg_input = 100. * input[:, 0] + 50. * input[:, 1] + 0. * input[:, 2]
        return super(CHAEDSmoothL1Loss, self).forward(reg_input, target)
