import itertools
from typing import List

import torch
from torchvision.transforms import functional

import src.datamodules.utils.functional


class OneHotToPixelLabelling(object):
    def __call__(self, tensor):
        return src.datamodules.utils.functional.argmax_onehot(tensor)


class RightAngleRotation:
    def __init__(self, angle_list: List[int] = [0, 90, 180, 270]):
        self.angles = angle_list
        self.target_class = None

    def _update_target_class(self) -> None:
        self.target_class = torch.randint(low=0, high=len(self.angles), size=(1,), dtype=torch.long).item()

    def __call__(self, tensor):
        self._update_target_class()
        rotation_angle = self.angles[self.target_class]
        return functional.rotate(img=tensor, angle=rotation_angle)

