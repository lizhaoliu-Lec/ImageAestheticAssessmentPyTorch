import numbers
from typing import List

from torch.linalg import Tensor
from torchvision import transforms
from torchvision.transforms.functional import _get_image_size, crop, center_crop

import random

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
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    def _select_according_to_rand():
        rand = random.randint(0, 4)

        if rand == 0:
            return crop(img, 0, 0, crop_height, crop_width)
        elif rand == 1:
            return crop(img, 0, image_width - crop_width, crop_height, crop_width)
        elif rand == 2:
            return crop(img, image_height - crop_height, 0, crop_height, crop_width)
        elif rand == 3:
            return crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
        else:
            return center_crop(img, [crop_height, crop_width])

    return _select_according_to_rand()


@TransformFactory.register('Scale')
class Scale(transforms.Scale):
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
