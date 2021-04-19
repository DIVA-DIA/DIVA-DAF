"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import logging
import re
from pathlib import Path
from typing import List, Tuple

import torch.utils.data as data
from torch import is_tensor
from torchvision.transforms import ToTensor

from hisDBDataModule.util.misc import has_extension, pil_loader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class CroppedHisDBDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png
    """

    def __init__(self, path: Path,
                 is_test=False, image_transform=None, target_transform=None, twin_transform=None,
                 classes=None, use_mask_train_val=False, use_mask_test=False, **kwargs):
        """
        #TODO doc
        Parameters
        ----------
        path : string
            Path to dataset folder (train / val / test)
        classes :
        workers : int
        imgs_in_memory :
        crops_per_image : int
        crop_size : int
        image_transform : callable
        target_transform : callable
        twin_transform : callable
        loader : callable
            A function to load an image given its path.
        """

        # Init list
        self.classes = classes
        # self.crops_per_image = crops_per_image
        self.use_mask_train_val = use_mask_train_val
        self.use_mask_test = use_mask_test

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths_per_page = self.get_gt_data_paths(path)

        # TODO: make more fanzy stuff here
        # self.img_paths = [pair for page in self.img_paths_per_page for pair in page]

        self.num_samples = len(self.img_paths_per_page)
        if self.num_samples == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                path, ",".join(IMG_EXTENSIONS)))

        if self.is_test:
            pass

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def __getitem__(self, index):
        if self.is_test:
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)

        img, gt, boundary_mask = self.apply_transformation(data_img, gt_img)
        if self.use_mask_train_val:
            return img, gt, boundary_mask
        else:
            return img, gt

    def _get_test_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt, boundary_mask = self.apply_transformation(data_img, gt_img)

        if self.use_mask_test:
            return img, gt, boundary_mask, index
        else:
            return img, gt, index

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.img_paths_per_page[index][0])
        gt_img = pil_loader(self.img_paths_per_page[index][1])

        return data_img, gt_img

    def apply_transformation(self, img, gt):
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        Parameters
        ----------
        img: PIL image
            image data
        gt: PIL image
            ground truth image
        coordinates: tuple (int, int)
            coordinates where the sliding window should be cropped
        Returns
        -------
        tuple
            img and gt after transformations
        """
        if self.twin_transform is not None and not self.is_test:
            img, gt = self.twin_transform(img, gt)

        if self.image_transform is not None:
            # perform transformations
            img, gt = self.image_transform(img, gt)

        if not is_tensor(img):
            img = ToTensor()(img)
        if not is_tensor(gt):
            gt = ToTensor()(gt)

        border_mask = gt[0, :, :] != 0
        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt, border_mask

    @staticmethod
    def get_gt_data_paths(directory: Path) -> List[Tuple[Path, Path, str, str, Tuple[int, int]]]:
        """
        Structure of the folder

        directory/data/ORIGINAL_FILENAME/FILE_NAME_X_Y.png
        directory/gt/ORIGINAL_FILENAME/FILE_NAME_X_Y.png


        :param directory:
        :return: tuple
            (path_data_file, path_gt_file, original_image_name, (x, y))
        """
        paths = []
        directory = directory.expanduser()

        path_data_root = directory / 'data'
        path_gt_root = directory / 'gt'

        if not (path_data_root.is_dir() or path_gt_root.is_dir()):
            logging.error("folder data or gt not found in " + str(directory))

        for path_data_subdir in sorted(path_data_root.iterdir()):
            if not path_data_subdir.is_dir() and has_extension(path_data_subdir.name, IMG_EXTENSIONS):
                logging.warning("image file found in data root: " + str(path_data_subdir))
                continue

            path_gt_subdir = path_gt_root / path_data_subdir.stem
            assert path_gt_subdir.is_dir()

            for path_data_file, path_gt_file in zip(sorted(path_data_subdir.iterdir()),
                                                    sorted(path_gt_subdir.iterdir())):
                assert has_extension(path_data_file.name, IMG_EXTENSIONS) == \
                       has_extension(path_gt_file.name, IMG_EXTENSIONS), \
                       'get_gt_data_paths(): image file aligned with non-image file'

                if has_extension(path_data_file.name, IMG_EXTENSIONS) and has_extension(path_gt_file.name,
                                                                                        IMG_EXTENSIONS):
                    assert path_data_file.stem == path_gt_file.stem, \
                        'get_gt_data_paths(): mismatch between data filename and gt filename'
                    coordinates = re.compile(r'.+_x(\d+)_y(\d+)\.')
                    m = coordinates.match(path_data_file.name)
                    if m is None:
                        continue
                    x = int(m.group(1))
                    y = int(m.group(2))
                    # TODO check if we need x/y
                    paths.append((path_data_file, path_gt_file, path_data_subdir.stem, path_data_file.stem, (x, y)))

        return paths
