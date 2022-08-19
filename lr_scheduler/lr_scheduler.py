from tokenize import group
import torch.optim.lr_scheduler

from lr_scheduler import LRSchedulerFactory
import lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

@LRSchedulerFactory.register('ConstantLRScheduler')
class ConstantLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


@LRSchedulerFactory.register('MultiStepLR')
class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    ...


@LRSchedulerFactory.register('ExponentialLR')
class ExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    ...


@LRSchedulerFactory.register('StepLR')
class StepLR(torch.optim.lr_scheduler.StepLR):
    ...
@LRSchedulerFactory.register('WarmupLR')
class WarmupLR(_LRScheduler):
    """WarmupLR initialize the optimizer with a small learning
    rate which grows to max in ``warmup_epoch``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epoch(int): epochs to warmup the learning rate.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose(bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, warmup_steps,last_epoch=-1,verbose=False) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.warmup_steps=warmup_steps
        self.step_counts=0
        self.initiallr=[group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch,verbose)
        
    def get_lr(self):
        if self.step_counts<self.warmup_steps:
            self.step_counts+=1
            return[1e-8+group*self.step_counts/(self.warmup_steps) for group in self.initiallr]  
        else:
            return self.initiallr

           