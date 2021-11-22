import math
import random

import torch
from PIL import Image
from torchvision.transforms import Pad

from src.datamodules.DivaHisDB.utils import functional as F_custom


class ResizePad(object):
    """
    Perform resizing keeping the aspect ratio of the image --padding type: continuous (black).
    Expects PIL image and int value as target_size
    (It can be extended to perform other transforms on both PIL image and object boxes.)

    Example:
    target_size = 200
    # im: numpy array
    img = Image.fromarray(im.astype('uint8'), 'RGB')
    img = ResizePad(target_size)(img)
    """

    def __init__(self, target_size):
        self.target_size = target_size
        self.boxes = torch.Tensor([[0, 0, 0, 0]])

    def resize(self, img, boxes, size, max_size=1000):
        '''Resize the input PIL image to the given size.
        Args:
          img: (PIL.Image) image to be resized.
          boxes: (tensor) object boxes, sized [#ojb,4].
          size: (tuple or int)
            - if is tuple, resize image to the size.
            - if is int, resize the shorter side to the size while maintaining the aspect ratio.
          max_size: (int) when size is int, limit the image longer size to max_size.
                    This is essential to limit the usage of GPU memory.
        Returns:
          img: (PIL.Image) resized image.
          boxes: (tensor) resized boxes.
        '''
        w, h = img.size
        if isinstance(size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(size) / size_min
            if sw * size_max > max_size:
                sw = sh = float(max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = size
            sw = float(ow) / w
            sh = float(oh) / h
        return img.resize((ow, oh), Image.BILINEAR), \
               boxes * torch.Tensor([sw, sh, sw, sh])

    def random_crop(self, img, boxes):
        '''Crop the given PIL image to a random size and aspect ratio.
        A crop of random size of (0.08 to 1.0) of the original size and a random
        aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
        Args:
          img: (PIL.Image) image to be cropped.
          boxes: (tensor) object boxes, sized [#ojb,4].
        Returns:
          img: (PIL.Image) randomly cropped image.
          boxes: (tensor) randomly cropped boxes.
        '''
        success = False
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.56, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x = random.randint(0, img.size[0] - w)
                y = random.randint(0, img.size[1] - h)
                success = True
                break

        # Fallback
        if not success:
            w = h = min(img.size[0], img.size[1])
            x = (img.size[0] - w) // 2
            y = (img.size[1] - h) // 2

        img = img.crop((x, y, x + w, y + h))
        boxes -= torch.Tensor([x, y, x, y])
        boxes[:, 0::2].clamp_(min=0, max=w - 1)
        boxes[:, 1::2].clamp_(min=0, max=h - 1)
        return img, boxes

    def center_crop(self, img, boxes, size):
        '''Crops the given PIL Image at the center.
        Args:
          img: (PIL.Image) image to be cropped.
          boxes: (tensor) object boxes, sized [#ojb,4].
          size (tuple): desired output size of (w,h).
        Returns:
          img: (PIL.Image) center cropped image.
          boxes: (tensor) center cropped boxes.
        '''
        w, h = img.size
        ow, oh = size
        i = int(round((h - oh) / 2.))
        j = int(round((w - ow) / 2.))
        img = img.crop((j, i, j + ow, i + oh))
        boxes -= torch.Tensor([j, i, j, i])
        boxes[:, 0::2].clamp_(min=0, max=ow - 1)
        boxes[:, 1::2].clamp_(min=0, max=oh - 1)
        return img, boxes

    def random_flip(self, img, boxes):
        '''Randomly flip the given PIL Image.
        Args:
            img: (PIL Image) image to be flipped.
            boxes: (tensor) object boxes, sized [#ojb,4].
        Returns:
            img: (PIL.Image) randomly flipped image.
            boxes: (tensor) randomly flipped boxes.
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
        return img, boxes

    def resize_with_padding(self, img, target_size):
        img, boxes = self.resize(img, self.boxes, target_size, max_size=target_size)
        padding = (max(0, target_size - img.size[0]) // 2, max(0, target_size - img.size[1]) // 2)
        img = Pad(padding)(img)

        return img

    def __call__(self, img):
        img = self.resize_with_padding(img, self.target_size)
        return img


class OneHotToPixelLabelling(object):
    def __call__(self, tensor):
        return F_custom.argmax_onehot(tensor)


class OneHotEncoding(object):
    def __init__(self, class_encodings):
        self.class_encodings = class_encodings

    def __call__(self, gt):
        """
        Args:

        Returns:

        """
        return F_custom.gt_to_one_hot(gt, self.class_encodings)


class IntegerEncoding(object):
    def __init__(self, class_encodings):
        self.class_encodings = class_encodings

    def __call__(self, gt):
        """
        Args:

        Returns:

        """
        return F_custom.gt_to_int_encoding(gt, self.class_encodings)
