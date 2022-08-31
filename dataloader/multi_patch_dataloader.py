import numpy
import torch
from dataloader.factory import DataloaderFactory
from torch.utils.data.dataloader import DataLoader
import random


def multi_patch_collate(batch, shuffle=False):
    assert isinstance(batch[0][0], torch.Tensor)
    "check dataset or transform"
    num_patches = batch[0][0].size(0)
    input_tensor = torch.stack([x[0] for x in batch], dim=0)
    input_tensor = torch.reshape(input_tensor, (-1, input_tensor.size(2), input_tensor.size(3), input_tensor.size(4)))

    def _stack(index):
        index_list = [i for i in range(len(batch))]
        # distribution
        if isinstance(batch[0][index], list):
            return torch.cat([torch.tensor([batch[i][index]]).repeat(num_patches, ) for i in index_list])
        elif isinstance(batch[0][index], numpy.ndarray):
            return torch.cat([torch.tensor([batch[i][index]]).repeat(num_patches, ) for i in index_list])
        # classification
        elif isinstance(batch[0][index], int):
            return torch.cat([torch.tensor([batch[i][index]]).repeat(num_patches, ) for i in index_list])
        else:
            raise TypeError("label types:{} except list and int are not implemented yet".format(type(batch[0][1])))

    if len(batch[0]) == 3:
        info_tensor = [_stack(i + 1) for i in range(len(batch[0]) - 1)]
    else:
        info_tensor = _stack(1)
    if shuffle:
        shuffle_list = torch.randperm(len(batch) * num_patches)
        input_tensor = torch.stack(input_tensor[shuffle_list])
        info_tensor[-1] = torch.stack(info_tensor[-1][shuffle_list])
    output = [input_tensor, info_tensor]
    return output


def sequence_multi_patch_collate(batch):
    return multi_patch_collate(batch, False)


def random_multi_patch_collate(batch):
    return multi_patch_collate(batch, True)


@DataloaderFactory.register("MultiPatchDataloader")
class MultiPatchDataloader(DataLoader):
    def __init__(self, dataset, collate_fn: str, *args, **kwargs):
        if collate_fn == "sequence_multi_patch_collate":
            super().__init__(dataset=dataset, collate_fn=sequence_multi_patch_collate, *args, **kwargs)
        else:
            super().__init__(dataset=dataset, collate_fn=random_multi_patch_collate, *args, **kwargs)


@DataloaderFactory.register("Dataloader")
class Dataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(Dataloader, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    def test_multi_patch_dataloader():
        import torch
        import numpy
        from trainer import trainer
        from trainer import config_parser
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        config = config_parser.ConfigParser(config_path="dataloader/test_multi_patch.yaml")

        train_transforms = trainer.get_transforms(config.train_transforms)
        test_transforms = trainer.get_transforms(config.test_transforms)
        train_dataset, test_dataset = trainer.get_train_test_dataset(
            {'train_dataset': config.train_dataset, 'test_dataset': config.test_dataset},
            train_transforms=train_transforms,
            test_transforms=test_transforms)
        train_loader = MultiPatchDataloader(dataset=train_dataset, collate_fn="sequence_multi_patch_collate",
                                            batch_size=5,
                                            shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=5)
        for batch_data_tensor, target in tqdm(train_loader):
            print(batch_data_tensor.size())
            print(target)

            break
        for batch_data_tensor, target in tqdm(test_loader):
            print(batch_data_tensor.size())
            print(target)

            break


    test_multi_patch_dataloader()
