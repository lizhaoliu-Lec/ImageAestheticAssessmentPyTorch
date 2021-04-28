import torch.optim.lr_scheduler

from lr_scheduler import LRSchedulerFactory


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
