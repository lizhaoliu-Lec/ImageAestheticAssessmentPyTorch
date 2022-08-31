import numbers
from typing import List

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import center_crop
import random
import torch.nn as nn
from transform.factory import TransformFactory


def random_five_crop(img: Tensor, size: List[int]) -> Tensor:
    """Crop the given image into four corners and the central crop,
    and return one of them randomly!
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
         img (PIL Image or Tensor): randomly select from (tl, tr, bl, br, center)
            Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))

    def _select_according_to_rand():
        rand = random.randint(0, 4)

        if rand == 0:
            return img.crop((0, 0, crop_w, crop_h))
        elif rand == 1:
            return img.crop((w - crop_w, 0, w, crop_h))
        elif rand == 2:
            return img.crop((0, h - crop_h, crop_w, h))
        elif rand == 3:
            return img.crop((w - crop_w, h - crop_h, w, h))
        else:
            return center_crop(img, (crop_h, crop_w))

    return _select_according_to_rand()


@TransformFactory.register('Scale')
class Scale(transforms.Resize):
    def __init__(self, size=256, **kwargs):
        super(Scale, self).__init__(size=size, **kwargs)


@TransformFactory.register('RandomFiveCrop')
class RandomFiveCrop(transforms.RandomCrop):
    def __init__(self, size=224, **kwargs):
        super().__init__(size=size, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            img (PIL Image or Tensor): Randomly selected from the standard five crop.
        """
        return random_five_crop(img, self.size)


@TransformFactory.register('RandomHorizontalFlip')
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    pass


@TransformFactory.register('ToTensor')
class ToTensor(transforms.ToTensor):
    pass


@TransformFactory.register('Normalize')
class Normalize(transforms.Normalize):
    def __init__(self, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__(mean=mean, std=std, **kwargs)


@TransformFactory.register('CenterCrop')
class CenterCrop(transforms.CenterCrop):
    pass


@TransformFactory.register('CropFivePatches')
class CropFivePatches(transforms.FiveCrop):
    def __init__(self, size=224):
        super().__init__(size=size)


@TransformFactory.register('RandomCrop')
class RandomCrop(transforms.RandomCrop):
    def __init__(self, size=224, **kwargs) -> None:
        super().__init__(size=size, **kwargs)


@TransformFactory.register('CropPatches')
class CropPatches(transforms.RandomCrop):
    def __init__(self, size=224, num_patches=5) -> None:
        super().__init__(size=size)
        self.num_patches = num_patches
        self.to_tensor_layer = transforms.ToTensor()

    def forward(self, img):
        return [super(CropPatches, self).forward(img) for _ in range(self.num_patches)]


@TransformFactory.register('MultiPatchFlip')
class MultiPatch(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img):
        return [super(MultiPatch, self).forward(_) for _ in img]


@TransformFactory.register('MultiPatchToTensor')
class MultiPatchToPatch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        to_tensor_layer = transforms.ToTensor()
        img_tensor = torch.stack([to_tensor_layer(i) for i in img])
        return img_tensor


if __name__ == '__main__':
    def run_crop_crop_five_patch_to_tensor():
        from trainer import trainer
        from trainer import config_parser
        from torch.utils.data import DataLoader

        config = config_parser.ConfigParser(config_path="test_multi_patch.yaml")

        train_transforms = trainer.get_transforms(config.train_transforms)
        test_transforms = trainer.get_transforms(config.test_transforms)
        train_dataset, test_dataset = trainer.get_train_test_dataset(config.dataset,
                                                                     train_transforms=train_transforms,
                                                                     test_transforms=test_transforms)
        train_loader = DataLoader(dataset=train_dataset, batch_size=5,
                                  shuffle=True, pin_memory=True)
        train_loader_iter = iter(train_loader)
        batch_data = next(train_loader_iter)
        for single_data in batch_data:
            # to_image_tool = transforms.ToPILImage()
            # img = to_image_tool(single_data[0])
            # img.show()
            print("[normal dataloader] size of single data in batch:{}".format(single_data.size()))


    run_crop_crop_five_patch_to_tensor()
