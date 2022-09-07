import torch
import torch.nn as nn
from loss import LossFactory
import torch.nn.functional as f


class BaseAttentionLoss(nn.Module):
    def __init__(self, beta: int = 2):
        super().__init__()
        self.beta = beta

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        # CrossEntropyLoss expect raw input, but not softmax(input)
        cross_entropy_loss = f.cross_entropy(_input, target, reduction='none')

        softmax_input = f.softmax(_input, 1)
        one_hot_target = f.one_hot(target, num_classes=2).to(_input.device)
        weight = 1 - softmax_input[one_hot_target.bool()] ** self.beta
        loss = torch.mean(weight * cross_entropy_loss)
        return loss


@LossFactory.register('AdaptiveAttentionLoss')
class AdaptiveAttentionLoss(BaseAttentionLoss):
    def __init__(self, beta: int = 2):
        super().__init__(beta)

    def forward(self, _input: torch.Tensor, target):
        if type(target) == torch.Tensor:
            return f.cross_entropy(_input, target, reduction='mean')
        elif type(target) == list and len(target) == 2:
            index = target[0]
            assert type(index) == torch.Tensor
            "index must be tensor but got{}".format(type(index))
            label = target[1]
            loss = 0
            for ind in torch.unique(index):
                input_ind = _input[ind == index]
                label_ind = label[ind == index]
                loss += super().forward(input_ind, label_ind)
            loss /= torch.unique(index).shape[0]
            return loss


@LossFactory.register("AverageAttentionLoss")
class AverageAttentionLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()

    def forward(self, _input: torch.Tensor, target):
        if type(target) == torch.Tensor:
            return f.cross_entropy(_input, target, reduction='mean')
        elif type(target) == list and len(target) == 2:
            index = target[0]
            label = target[1]
            loss = 0
            for ind in torch.unique(index):
                input_ind = _input[ind == index]
                label_ind = label[ind == index]
                loss += super().forward(input_ind, label_ind)
            loss /= torch.unique(index).shape[0]
            return loss


# min_probability=max_loss
def max_attention_loss(_input, target):
    cross_entropy_loss = f.cross_entropy(_input, target, reduction='none')
    return torch.max(cross_entropy_loss)


@LossFactory.register("MaxAttentionLoss")
class MaxAttentionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input, target):
        if type(target) == torch.Tensor:
            return f.cross_entropy(_input, target, reduction='mean')
        elif type(target) == list and len(target) == 2:
            index = target[0]
            label = target[1]
            loss = 0
            for ind in torch.unique(index):
                input_ind = _input[ind == index]
                label_ind = label[ind == index]
                loss += max_attention_loss(input_ind, label_ind)
            loss /= torch.unique(index).shape[0]
            return loss


if __name__ == '__main__':
    def adaptive_attention_loss_with_index(sample_input, sample_target):
        attention_loss = AdaptiveAttentionLoss(2)
        loss = attention_loss.forward(sample_input, sample_target)
        normal_cross_entropy_loss_layer = nn.CrossEntropyLoss()
        normal_cross_entropy_loss = normal_cross_entropy_loss_layer.forward(input=sample_input, target=sample_target[1])
        print('normal loss{}'.format(normal_cross_entropy_loss))
        print('adaptive attention based cross entropy loss group by image{}'.format(loss))


    def max_attention_loss_with_index(sample_input, sample_target):
        attention_loss = MaxAttentionLoss()
        loss = attention_loss.forward(sample_input, sample_target)
        normal_cross_entropy_loss_layer = nn.CrossEntropyLoss()
        normal_cross_entropy_loss = normal_cross_entropy_loss_layer.forward(input=sample_input, target=sample_target[1])
        print('normal loss{}'.format(normal_cross_entropy_loss))
        print('max attention based cross entropy loss group by image{}'.format(loss))


    def average_attention_loss_with_index(sample_input, sample_target):
        attention_loss = AverageAttentionLoss()
        loss = attention_loss.forward(sample_input, sample_target)
        normal_cross_entropy_loss_layer = nn.CrossEntropyLoss()
        normal_cross_entropy_loss = normal_cross_entropy_loss_layer.forward(input=sample_input, target=sample_target[1])
        print('normal loss{}'.format(normal_cross_entropy_loss))
        print('average attention based cross entropy loss group by image{}'.format(loss))


    def generate_test_data():
        sample_input = torch.rand(25, 2) * 7
        sample_index = torch.repeat_interleave(torch.tensor([0, 1, 2, 3, 4]), 5)
        sample_label = (torch.rand(5) > 0.5).long()
        sample_label = torch.repeat_interleave(sample_label, 5)
        sample_target = [sample_index, sample_label]
        return sample_input, sample_target


    # adaptive_attention_loss()
    # adaptive_attention_loss_with_index()

    def show_all_attention_loss():
        sample_input, sample_target = generate_test_data()
        adaptive_attention_loss_with_index(sample_input, sample_target)
        average_attention_loss_with_index(sample_input, sample_target)
        max_attention_loss_with_index(sample_input, sample_target)


    def gpu_loss():
        import torch
        from trainer import trainer
        from trainer import config_parser
        import dataloader as dl
        from tqdm import tqdm

        config = config_parser.ConfigParser(config_path="../dataloader/test_multi_patch.yaml")

        train_transforms = trainer.get_transforms(config.train_transforms)
        test_transforms = trainer.get_transforms(config.test_transforms)
        train_dataset, test_dataset = trainer.get_train_test_dataset(
            {'train_dataset': config.train_dataset, 'test_dataset': config.test_dataset},
            train_transforms=train_transforms,
            test_transforms=test_transforms)
        train_loader = dl.MultiPatchDataloader(dataset=train_dataset,
                                               collate_fn="sequence_multi_patch_collate",
                                               batch_size=5,
                                               shuffle=True, pin_memory=True)

        for batch_data_tensor, target in tqdm(train_loader):
            print(batch_data_tensor)
            print(target[0])
            print(target[1])
            attention_loss = AdaptiveAttentionLoss(2)
            loss = attention_loss(batch_data_tensor, target)
            print(loss)


    def show_test_loss():
        sample_input, sample_target = generate_test_data()
        attention_loss = AdaptiveAttentionLoss(2)
        loss = attention_loss.forward(sample_input, sample_target[-1])
        print("test loss {}".format(loss))
        adaptive_attention_loss_with_index(sample_input, sample_target)


    # gpu_loss()

    show_all_attention_loss()
    # show_test_loss()
