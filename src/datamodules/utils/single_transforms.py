import itertools
from typing import List, Tuple

import math
import numpy as np
import torch
from torchvision.transforms import functional

import src.datamodules.utils.functional


class OneHotToPixelLabelling(object):
    def __call__(self, tensor):
        return src.datamodules.utils.functional.argmax_onehot(tensor)


class RightAngleRotation:
    def __init__(self, angle_list: List[int] = [0, 90, 180, 270]):
        self.angles = angle_list
        self.target_class = torch.randint(low=0, high=len(self.angles), size=(1,))

    def _update_target_class(self) -> None:
        self.target_class = torch.randint(low=0, high=len(self.angles), size=(1,))

    def __call__(self, tensor):
        rotation_angle = self.angles[self.target_class]
        self._update_target_class()
        return functional.rotate(img=tensor, angle=rotation_angle)


class TilesBuilding:
    def __init__(self, rows: int, cols: int, fixed_positions: int = 0, width_center_crop: int = 840,
                 height_center_crop: int = 1200):
        self.rows = rows
        self.cols = cols
        self.permutations = np.array(sorted(list(itertools.permutations(range(rows * cols)))))
        self.classes = np.array(list(range(rows * cols)))
        self.fixed_positions = fixed_positions
        self.filtered_perms = self._get_perms_with_n_fixed_positions()
        self.target_class = np.random.randint(low=0, high=len(self.filtered_perms), size=(1,))
        self.width_center_crop = width_center_crop
        self.height_center_crop = height_center_crop
        self.width_tile = self.width_center_crop // self.cols
        self.height_tile = self.height_center_crop // self.rows

    def _update_target_perm(self) -> None:
        self.target_class = np.random.randint(low=0, high=len(self.filtered_perms), size=(1,))

    def _get_perms_with_n_fixed_positions(self) -> np.ndarray:
        return np.array([p for p in self.permutations if np.sum(p == self.classes) == self.fixed_positions])

    def __call__(self, tensor):
        self._update_target_perm()
        return self._get_tile_image(tensor, torch.tensor(self.filtered_perms[self.target_class]))

    def _get_tile_image(self, current_img: torch.Tensor, permutation: torch.Tensor):
        cropped_img = functional.center_crop(current_img, [self.width_center_crop, self.height_center_crop])
        permutation = permutation.reshape((self.cols, self.rows))
        new_img_tensor = current_img.clone()
        width_offset = ((current_img.shape[1] - cropped_img.shape[1]) // 2)
        height_offset = ((current_img.shape[0] - cropped_img.shape[0]) // 2)

        random_width_offset = torch.randint(-3, 3, (1,))
        random_height_offset = torch.randint(-3, 3, (1,))

        for i in range(self.rows):
            for j in range(self.cols):
                w_begin_crop = ((permutation[i, j] % self.cols) * self.width_tile)
                w_stop_crop = w_begin_crop + self.width_tile
                h_begin_crop = ((permutation[i, j] // self.cols) * self.height_tile)
                h_stop_crop = h_begin_crop + self.height_tile

                width_start_ori = width_offset + (j * self.width_tile) + random_width_offset
                width_end_ori = width_offset + ((j + 1) * self.width_tile) + random_width_offset
                height_start_ori = height_offset + (i * self.height_tile) + random_height_offset
                height_end_ori = height_offset + ((i + 1) * self.height_tile) + random_height_offset

                new_img_tensor[height_start_ori: height_end_ori, width_start_ori: width_end_ori, :] = cropped_img[
                     h_begin_crop:h_stop_crop,
                     w_begin_crop:w_stop_crop]

        return new_img_tensor
