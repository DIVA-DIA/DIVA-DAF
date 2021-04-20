"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import itertools
import logging
import math
import os
import os.path
import sys

import torch.utils.data as data
from torchvision.transforms import ToTensor

from src.datamodules.hisDBDataModule.util.misc import has_extension, pil_loader
from src.datamodules.hisDBDataModule.util.transformations.transforms import ToTensorSlidingWindowCrop

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']





def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'PIL':
        return pil_loader(path)
    else:
        logging.info("Something went wrong with the default_loader in image_folder_segmentation.py")
        sys.exit(-1)


class ImageFolderSegmentationDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png
    """

    def __init__(self, path, workers, imgs_in_memory, crops_per_image, crop_size,
                 is_test=False, transform=None, target_transform=None, twin_transform=None,
                 loader=default_loader, classes=None, **kwargs):
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
        transform : callable
        target_transform : callable
        twin_transform : callable
        loader : callable
            A function to load an image given its path.
        """

        # Init list
        self.root = str(path)
        self.classes = classes
        self.num_workers = workers
        # self.num_workers = 1 if is_test else workers
        self.imgs_in_memory = imgs_in_memory
        self.crops_per_image = crops_per_image
        self.crop_size = crop_size
        # transformations
        self.transform = transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.loader = loader
        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths = self.get_gt_data_paths(path)
        self.num_imgs_in_set = len(self.img_paths)
        if self.num_imgs_in_set == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                path, ",".join(IMG_EXTENSIONS)))

        self.current_img_index = -1

        if self.is_test:
            # Overlap for the sliding window (% of crop size)
            self.overlap = 0.5
            # Get the numbers for __len__
            self.img_names_sizes, self.num_horiz_crops, self.num_vert_crops = self._get_img_size_and_crop_numbers()
            self.test_crop_list = self._get_test_crop_list()

        # Make sure work can be split into workers equally
        if self.num_workers > 1:
            # Length is divisible by the number of workers
            if self.__len__() % self.num_workers != 0:
                logging.error("{} (number of pages in set ({}) * crops per image {}) "
                              "must be divisible by the number of workers (currently {})".format(
                    self.__len__(), self.num_imgs_in_set, self.crops_per_image, self.num_workers))
                sys.exit(-1)
            # Crops per page is divisible by the number of workers
            if self.crops_per_image % self.num_workers != 0:
                logging.error("{} (# crops per page) must be divisible by the number of"
                              " workers (currently {})".format(self.crops_per_image, self.num_workers))
                sys.exit(-1)

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        if self.is_test:
            # Sliding window
            return len(self.test_crop_list)
            # return sum([hc * vc for hc, vc in zip(self.num_horiz_crops, self.num_vert_crops)])
        else:
            # Number of images in the dataset * how many crops per page
            return len(self.img_paths) * self.crops_per_image

    def __getitem__(self, index):
        # TODO documentation format
        """
        Args:
            index (int): Index

        Returns:
            during train and val
            tuple: (img_input, gt_target)

            during test
            tuple: ((img_input, orig_img_shape, top_left_coordinates_of_crop, test_img_name), gt_target)
                default sliding window (used during test)
        """
        if self.is_test:
            # Testing: sliding window with overlap
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index):
        img_index = int(index / self.crops_per_image)
        if img_index > len(self.img_paths):
            print(f'img_index out of range: {img_index} = {index} % {self.crops_per_image} :: len(self.img_paths): {len(self.img_paths)}')
        self._load_image_and_var(img_index=img_index)

        img, gt, boundary_mask = self.apply_transformation(self.current_data_img, self.current_gt_img)

        return img, (gt, boundary_mask)

    def _get_test_items(self, index):
        # TODO documentation

        img_index, x, y = self.test_crop_list[index]
        self._load_image_and_var(img_index=img_index)
        coordinates = (x, y)

        img, gt, boundary_mask = self.apply_transformation(self.current_data_img, self.current_gt_img,
                                                           coordinates=coordinates)

        return (img, coordinates, img_index), (gt, boundary_mask)

    def _load_image_and_var(self, img_index):
        """
        Inits the variables responsible of tracking which crop should be taken next, the current images and the like.
        This should be run every time a new page gets loaded for the test-set
        """

        if self.current_img_index == img_index:
            return

        # Load image
        self.current_data_img = pil_loader(self.img_paths[img_index][0])
        self.current_gt_img = pil_loader(self.img_paths[img_index][1])

        # Update pointer to current image
        self.current_img_index = img_index

    def apply_transformation(self, img, gt, coordinates=None):
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

        if self.transform is not None:
            # perform transformations
            img, gt = self.transform(img, gt)

        # convert to tensor
        if self.is_test:
            # crop for sliding window
            img, gt = ToTensorSlidingWindowCrop(self.crop_size)(img, gt, coordinates)
        else:
            img, gt = ToTensor()(img), ToTensor()(gt)

        border_mask = gt[0, :, :] != 0
        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt, border_mask

    def _get_img_size_and_crop_numbers(self):
        # TODO documentation
        img_names_sizes = []  # list of tuples -> (gt_img_name, img_size (H, W))
        num_horiz_crops = []
        num_vert_crops = []

        for img_path, gt_path in self.img_paths:
            data_img = self.loader(img_path)
            gt_img = self.loader(gt_path)
            # Ensure that data and gt image are of the same size
            assert gt_img.size == data_img.size
            img_names_sizes.append((os.path.basename(gt_path), data_img.size[::-1]))
            step_size = self.crop_size * self.overlap
            num_horiz_crops.append(math.ceil((data_img.size[1] - self.crop_size) / step_size + 1))
            num_vert_crops.append(math.ceil((data_img.size[0] - self.crop_size) / step_size + 1))

        return img_names_sizes, num_horiz_crops, num_vert_crops

    def _get_test_crop_list(self):
        return [self._convert_crop_id_to_coordinates(img_index, hcrop_index, vcrop_index) for img_index in
                range(self.num_imgs_in_set) for hcrop_index, vcrop_index in
                itertools.product(range(self.num_horiz_crops[img_index]),
                                  range(self.num_vert_crops[img_index]))]

    def _convert_crop_id_to_coordinates(self, img_index, hcrop_index, vcrop_index):
        # X coordinate
        if hcrop_index == self.num_horiz_crops[img_index] - 1:
            # We are at the end of a line
            x_position = self.img_names_sizes[img_index][1][0] - self.crop_size
        else:
            x_position = int(self.crop_size / (1 / self.overlap)) * hcrop_index
            assert x_position < self.img_names_sizes[img_index][1][0] - self.crop_size

        # Y coordinate
        if vcrop_index == self.num_vert_crops[img_index] - 1:
            # We are at the bottom end
            y_position = self.img_names_sizes[img_index][1][1] - self.crop_size
        else:
            y_position = int(self.crop_size / (1 / self.overlap)) * vcrop_index
            assert y_position < self.img_names_sizes[img_index][1][1] - self.crop_size

        return img_index, x_position, y_position

    @staticmethod
    def get_gt_data_paths(directory):
        """
        Parameters
        ----------
        directory: string
            parent directory with gt and data folder inside

        Returns
        -------
        paths: list of tuples

        """

        paths = []
        directory = os.path.expanduser(directory)

        path_imgs = os.path.join(directory, "data")
        path_gts = os.path.join(directory, "gt")

        if not (os.path.isdir(path_imgs) or os.path.isdir(path_gts)):
            logging.error("folder data or gt not found in " + str(directory))

        for img_name, gt_name in zip(sorted(os.listdir(path_imgs)), sorted(os.listdir(path_gts))):
            assert has_extension(img_name, IMG_EXTENSIONS) == has_extension(gt_name, IMG_EXTENSIONS), \
                'get_gt_data_paths(): image file aligned with non-image file'

            if has_extension(img_name, IMG_EXTENSIONS) and has_extension(gt_name, IMG_EXTENSIONS):
                assert os.path.splitext(img_name)[0] == os.path.splitext(gt_name)[0], \
                    'get_gt_data_paths(): mismatch between data filename and gt filename'
                paths.append((os.path.join(path_imgs, img_name), os.path.join(path_gts, gt_name)))

        return paths