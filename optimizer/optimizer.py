import torch.optim

from optimizer.factory import OptimizerFactory


@OptimizerFactory.register('SGD')
class SGD(torch.optim.SGD):
    ...


@OptimizerFactory.register('Adam')
class Adam(torch.optim.Adam):
    ...
