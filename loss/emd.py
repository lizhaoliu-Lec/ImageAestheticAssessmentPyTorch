"""
This is the implementation of earth mover’s distance,
which is first used in the field of image aesthetic assessment by
NIMA: Neural Image Assessment TIP 2018: https://arxiv.org/pdf/1709.05424

Note that we are not using the original implementation based on for-loop to iterate over the class dimension.
Instead, we use the torch.cumsum api to re-implement it. Note that the original implementation is also copied below
for testing the consistency between two implementations.
"""

import torch
import torch.nn as nn

from loss.factory import LossFactory


def single_emd_loss(p, q, r=2):
    """
    Implementation from https://github.com/kentsyx/Neural-IMage-Assessment
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Implementation from https://github.com/kentsyx/Neural-IMage-Assessment
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def earth_mover_distance(input: torch.Tensor, target: torch.Tensor, r: float = 2):
    """
    Batch Earth Mover's Distance implementation.
    Args:
        input: B x num_classes
        target: B x num_classes
        r: float, to penalize the Euclidean distance between the CDFs

    Returns:

    """
    N, num_classes = input.size()
    input_cumsum = torch.cumsum(input, dim=-1)
    target_cumsum = torch.cumsum(target, dim=-1)

    diff = torch.abs(input_cumsum - target_cumsum) ** r

    class_wise = (torch.sum(diff, dim=-1) / num_classes) ** (1. / r)
    scalar_ret = torch.sum(class_wise) / N
    return scalar_ret


@LossFactory.register('EarthMoverDistanceLoss')
class EarthMoverDistanceLoss(nn.Module):
    def __init__(self, r: float = 2.0):
        super().__init__()
        self.r = r

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return earth_mover_distance(input, target, r=self.r)



if __name__ == '__main__':
    def run_earth_mover_distance():
        a = torch.softmax(torch.randn((5, 100)), dim=-1)
        b = torch.softmax(torch.randn((5, 100)), dim=-1)
        ret_emd = emd_loss(a.clone(), b.clone())
        ret = earth_mover_distance(a.clone(), b.clone())
        print("===> ret for ori implementation: ", ret_emd)
        print("===> ret for cumsum implementation: ", ret)


    run_earth_mover_distance()
