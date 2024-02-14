import itertools

import numpy as np
import torch
from torchvision.transforms import functional


class TilesBuilding:
    def __init__(self, rows: int, cols: int, fixed_positions: int = 0, width_center_crop: int = 840,
                 height_center_crop: int = 1200):
        self.rows = rows
        self.cols = cols
        self.permutations = np.array(sorted(list(itertools.permutations(range(rows * cols)))))
        self.classes = np.array(list(range(rows * cols)))
        self.fixed_positions = fixed_positions
        self.filtered_perms = self._get_perms_with_n_fixed_positions()
        self.target_class = None
        self.width_center_crop = width_center_crop
        self.height_center_crop = height_center_crop
        self.width_tile = self.width_center_crop // self.cols
        self.height_tile = self.height_center_crop // self.rows
        self._check_values()

    def _check_values(self):
        if self.rows * self.cols - 1 == self.fixed_positions:
            raise ValueError("The number of fixed positions must be less than the number of tiles - 1")
        if self.rows * self.cols <= self.fixed_positions:
            raise ValueError("The number of fixed positions must be less than the number of tiles")

    def _update_target_perm(self) -> None:
        self.target_class = np.random.randint(low=0, high=len(self.filtered_perms), size=(1,))

    def _get_perms_with_n_fixed_positions(self) -> np.ndarray:
        return np.array([p for p in self.permutations if np.sum(p == self.classes) == self.fixed_positions])

    def __call__(self, tensor):
        self._update_target_perm()
        return self._get_tile_image(tensor, torch.tensor(self.filtered_perms[self.target_class]))

    def _get_tile_image(self, current_img: torch.Tensor, permutation: torch.Tensor):
        cropped_img = functional.center_crop(current_img, [self.height_center_crop, self.width_center_crop])
        permutation = permutation.reshape((self.rows, self.cols))
        new_img_tensor = current_img.clone()
        w_offset = ((current_img.shape[2] - cropped_img.shape[2]) // 2)
        h_offset = ((current_img.shape[1] - cropped_img.shape[1]) // 2)

        random_width_offset = torch.randint(-3, 3, (1,))
        random_height_offset = torch.randint(-3, 3, (1,))

        for i in range(self.rows):
            for j in range(self.cols):
                w_begin_crop = ((permutation[i, j] % self.cols) * self.width_tile)
                w_stop_crop = w_begin_crop + self.width_tile
                h_begin_crop = ((permutation[i, j] // self.cols) * self.height_tile)
                h_stop_crop = h_begin_crop + self.height_tile

                w_begin_o = w_offset + (j * self.width_tile) + random_width_offset
                w_stop_o = w_offset + ((j + 1) * self.width_tile) + random_width_offset
                h_begin_o = h_offset + (i * self.height_tile) + random_height_offset
                h_stop_o = h_offset + ((i + 1) * self.height_tile) + random_height_offset

                new_img_tensor[:, h_begin_o: h_stop_o, w_begin_o: w_stop_o] = cropped_img[:,
                                                                                 h_begin_crop:h_stop_crop,
                                                                                 w_begin_crop:w_stop_crop]

        return new_img_tensor
