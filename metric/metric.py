import torch
import torch.nn.functional as F

from metric.factory import MetricFactory


@MetricFactory.register('Accuracy')
class Accuracy:
    @torch.no_grad()
    def __call__(self, prediction, target):
        return torch.mean((torch.argmax(prediction, dim=1) == target).float()).item()


@MetricFactory.register('MAE')
class MAE:
    @torch.no_grad()
    def __call__(self, prediction, target):
        return F.l1_loss(prediction, target).item()


if __name__ == '__main__':
    def run_accuracy():
        prediction = torch.randn((1000, 10)).cuda()
        target = torch.randint(0, 10, (1000,)).cuda()
        print("===> acc: ", Accuracy()(prediction, target))


    def run_mse():
        prediction = torch.randn((2,)).cuda()
        target = torch.randn((2,)).cuda()
        print("==> prediction: ", prediction)
        print("==> target: ", target)
        print("===> mse: ", MAE()(prediction, target))


    # run_accuracy()
    run_mse()
