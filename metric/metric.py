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
        # for maximize the metric
        return -1. * F.l1_loss(prediction, target).item()


@MetricFactory.register('CHAEDMAE')
class CHAEDMAE:
    @torch.no_grad()
    def __call__(self, prediction, target):
        # for maximize the metric
        # input: (N, 3), target: (N)
        assert prediction.size()[1] == 3
        reg_prediction = 100. * prediction[:, 0] + 50. * prediction[:, 1] + 0. * prediction[:, 2]
        return -1. * F.l1_loss(reg_prediction, target).item()


@MetricFactory.register('MSE')
class MSE:
    @torch.no_grad()
    def __call__(self, prediction, target):
        # for maximize the metric
        return -1. * F.mse_loss(prediction, target).item()


@MetricFactory.register('AccuracyFromDistribution')
class AccuracyFromDistribution:
    def __init__(self, cut_off=5.0):
        self.cut_off = cut_off

    @staticmethod
    def get_score_from_distribution(distribution: torch.Tensor):
        # distribution: (N, num_classes)
        N, num_classes = distribution.size()
        arrange_index = torch.stack([torch.arange(1, num_classes + 1) for _ in range(N)], dim=0)
        return torch.sum(distribution * arrange_index.float().to(distribution.device), dim=-1)

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction_label = self.get_score_from_distribution(prediction) > self.cut_off
        target_label = self.get_score_from_distribution(target) > self.cut_off
        return torch.mean((prediction_label == target_label).float()).item()


@MetricFactory.register('AccuracyFromDistributionV2')
class AccuracyFromDistributionV2:
    def __init__(self):
        pass

    @staticmethod
    def get_score_from_distribution(distribution: torch.Tensor):
        # distribution: (N, num_classes)
        N, num_classes = distribution.size()
        return torch.argmax(distribution, 1)

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction_label = self.get_score_from_distribution(prediction)
        target_label = self.get_score_from_distribution(target)
        return torch.mean((prediction_label == target_label).float()).item()


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


    def run_acc_from_distribution():
        prediction = torch.softmax(torch.randn(5, 10), dim=-1)
        target = torch.softmax(torch.randn(5, 10), dim=-1)
        print("==> prediction: ", prediction)
        print("==> target: ", target)
        print("===> accuracy from distribution: ", AccuracyFromDistribution()(prediction, target))


    # run_accuracy()
    # run_mse()
    run_acc_from_distribution()
