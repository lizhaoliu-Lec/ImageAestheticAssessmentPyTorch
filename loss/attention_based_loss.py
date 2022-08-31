import torch
import torch.nn as nn
from loss import LossFactory
import torch.nn.functional as f


@LossFactory.register('AdaptiveAttentionBasedLoss')
class AttentionLoss(nn.Module):
    def __init__(self, r: float = 2.0, b: int = 2):
        super().__init__()
        self.r = r
        self.b = b

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        # CrossEntropyLoss expect raw input, but not softmax(input)
        cross_entropy_loss = f.cross_entropy(_input, target, reduction='none')

        softmax_input = f.softmax(_input, 1)
        one_hot_target = f.one_hot(target, num_classes=2)
        weight = torch.ones(softmax_input.size(0)) - torch.sum(one_hot_target * softmax_input, dim=1) ** self.b
        if __name__ == '__main__':
            print("before weight loss :{}".format(cross_entropy_loss))
            print("Attention weight{}".format(weight))
        #
        return torch.mean(weight * cross_entropy_loss)


if __name__ == '__main__':
    sample_input = torch.rand(5, 2) * 7
    sample_target = torch.rand(5) > 0.5
    sample_target = sample_target.long()
    attention_loss = AttentionLoss(2.0, 2)
    loss = attention_loss.forward(sample_input, sample_target)
    normal_cross_entropy_loss_layer = nn.CrossEntropyLoss()
    normal_cross_entropy_loss = normal_cross_entropy_loss_layer.forward(input=sample_input, target=sample_target)
    print('normal loss{}'.format(normal_cross_entropy_loss))
    print('attention based cross entropy loss{}'.format(loss))

