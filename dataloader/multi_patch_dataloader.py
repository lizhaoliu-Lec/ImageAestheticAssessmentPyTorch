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
    index_list = [i for i in range(len(batch))]
    if shuffle:
        random.shuffle(index_list)
    # distribution
    if isinstance(batch[0][1], list):
        label_tensor = torch.cat([torch.tensor([batch[i][1]]).repeat(num_patches, 1) for i in index_list])
    elif isinstance(batch[0][1], numpy.ndarray):
        label_tensor = torch.cat([torch.tensor([batch[i][1]]).repeat(num_patches, 1) for i in index_list])
    # classification
    elif isinstance(batch[0][1], int):
        label_tensor = torch.cat([torch.tensor([batch[i][1]]).repeat(num_patches, 1) for i in index_list])
    else:
        raise TypeError("label types:{} except list and int are not implemented ".format(type(batch[0][1])))
    output = [input_tensor, label_tensor]
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
    import torch
    import numpy
    from trainer import trainer
    from trainer import config_parser
    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter

    config = config_parser.ConfigParser(config_path="dataloader/test_multi_patch.yaml")

    train_transforms = trainer.get_transforms(config.train_transforms)
    test_transforms = trainer.get_transforms(config.test_transforms)
    train_dataset, test_dataset = trainer.get_train_test_dataset(config.dataset, train_transforms=train_transforms,
                                                                 test_transforms=test_transforms)
    train_loader = MultiPatchDataloader(dataset=train_dataset, collate_fn="sequence_multi_patch_collate", batch_size=5,
                                        shuffle=True, pin_memory=True)
    train_loader_iter = iter(train_loader)
    batch_data_tensor = next(train_loader_iter)
    input_data = batch_data_tensor[0]
    label = batch_data_tensor[1]
    print("size of input data in a batch{}".format(input_data.size()))
    print("size of label in a batch{}".format(label.size()))
    writer = SummaryWriter("dataloader/log")
    for i in range(5 * 5):
        img_array = numpy.array(input_data[i])
        writer.add_image("train", img_array, i, dataformats="CHW")
        print("{} score: {}".format(i,label[i]))
