import itertools
from typing import List, Tuple

import numpy as np
import torch
from torchvision.transforms import functional
from torch.nn.functional import unfold

import src.datamodules.utils.functional


class OneHotToPixelLabelling(object):
    """
    Transforms a one-hot encoded tensor to a pixel labelling tensor.

    :param tensor: The one-hot encoded tensor
    :type tensor: torch.Tensor
    :returns: The pixel labelling tensor
    :rtype: torch.Tensor
    """

    def __call__(self, tensor: torch.Tensor):
        return src.datamodules.utils.functional.argmax_onehot(tensor)


class RightAngleRotation:
    """
    Rotates the input tensor by a random angle from the list of angles.
    To also get the class if this is used in a gt generation context, the class is accessible via .target_class.

    :param angle_list: The list of angles to choose from
    :type angle_list: List[int]

    """

    def __init__(self, angle_list=None):
        if angle_list is None:
            angle_list = [0, 90, 180, 270]
        self.angles = angle_list
        self.target_class = None

    def _update_target_class(self) -> None:
        """
        Updates the target class to a random class from the list of angles.
        """
        self.target_class = torch.randint(low=0, high=len(self.angles), size=(1,), dtype=torch.long).item()

    def __call__(self, tensor):
        """
        Rotates the input tensor by the angle of the target class.

        :param tensor: to turn
        :type tensor: torch.Tensor
        :return: the rotated tensor (class is accessible via .target_class)
        :rtype: torch.Tensor
        """
        self._update_target_class()
        rotation_angle = self.angles[self.target_class]
        return functional.rotate(img=tensor, angle=rotation_angle)


class TilesBuilding:
    """
    Applies the idea of an embedded jigsaw puzzle on an image. The image is divided into a grid of tiles and then the
    tiles are shuffled. The number of rows and columns of the grid, the number of fixed positions and the size of the
    center crop can be set. The number of fixed positions must be less than the number of tiles - 1 and less than the
    number of tiles.
    To also get the class if this is used in a gt generation context, the class is accessible via .target_class.

    :param rows: The number of rows of the grid
    :type rows: int
    :param cols: The number of columns of the grid
    :type cols: int
    :param fixed_positions: The number of fixed positions in the grid
    :type fixed_positions: int
    :param width_center_crop: The width of the center crop
    :type width_center_crop: int
    :param height_center_crop: The height of the center crop

    """

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


class MorphoBuilding:
    """
    Applies the idea of morphological operators to build the GT.

    Code inspiration from: https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate

    """

    def __init__(self, dilation_filter_size: Tuple[int], erosion_filter_size: Tuple[int], border_size: int):
        self.dilation_filter_size = torch.tensor(dilation_filter_size)
        self.erosion_filter_size = torch.tensor(erosion_filter_size)
        self.border_size = border_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.morpho(tensor=tensor)

    def morpho(self, tensor: torch.Tensor) -> torch.Tensor:
        return

    def _dilation(self, image):
        # first pad the image to have correct unfolding; here is where the origins is used
        image_pad = functional.pad(img=image, padding=[0, self.dilation_filter_size[0] - 1, 0, self.dilation_filter_size.shape[1] - 1],
                                   padding_mode='constant', fill=0)
        # Unfold the image to be able to perform operation on neighborhoods
        image_unfold = unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=self.dilation_filter_size.shape)
        # Flatten the structural element since its two dimensions have been flatten when unfolding
        strel_flatten = torch.flatten(self.dilation_filter_size).unsqueeze(0).unsqueeze(-1)
        # Perform the greyscale operation; sum would be replaced by rest if you want erosion
        sums = image_unfold + strel_flatten
        # Take maximum over the neighborhood
        result, _ = sums.max(dim=1)
        # Reshape the image to recover initial shape
        return torch.reshape(result, image.shape)
