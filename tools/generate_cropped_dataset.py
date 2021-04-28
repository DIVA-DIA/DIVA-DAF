"""
Load a dataset of historic documents by specifying the folder where its located.
"""

import argparse
# Utils
import itertools
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def has_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py.

    Parameters
    ----------
    filename : string
        path to a file
    extensions : list
        extensions to match against
    Returns
    -------
    bool
        True if the filename ends with one of given extensions, false otherwise.
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def pil_loader(path, to_rgb=True):
    pic = Image.open(path)
    if to_rgb:
        pic = convert_to_rgb(pic)
    return pic


def convert_to_rgb(pic):
    if pic.mode == "RGB":
        pass
    elif pic.mode in ("CMYK", "RGBA", "P"):
        pic = pic.convert('RGB')
    elif pic.mode == "I":
        img = (np.divide(np.array(pic, np.int32), 2 ** 16 - 1) * 255).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    elif pic.mode == "I;16":
        img = (np.divide(np.array(pic, np.int16), 2 ** 8 - 1) * 255).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    elif pic.mode == "L":
        img = np.array(pic).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    else:
        raise TypeError(f"unsupported image type {pic.mode}")
    return pic


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
    directory = Path(directory).expanduser()

    path_imgs = Path(directory) / "data"
    path_gts = Path(directory) / "gt"

    if not (path_imgs.is_dir() or path_gts.is_dir()):
        logging.error("folder data or gt not found in " + str(directory))

    for img_name, gt_name in zip(sorted(path_imgs.iterdir()), sorted(path_gts.iterdir())):
        assert has_extension(str(img_name), IMG_EXTENSIONS) == has_extension(str(gt_name), IMG_EXTENSIONS), \
            'get_gt_data_paths(): image file aligned with non-image file'

        if has_extension(str(img_name), IMG_EXTENSIONS) and has_extension(str(gt_name), IMG_EXTENSIONS):
            assert img_name.suffix[0] == gt_name.suffix[0], \
                'get_gt_data_paths(): mismatch between data filename and gt filename'
            paths.append((path_imgs / img_name, path_gts / gt_name))

    return paths


class ToTensorSlidingWindowCrop(object):
    """
    Crop the data and ground truth image at the specified coordinates to the specified size and convert
    them to a tensor.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, gt, coordinates):
        """
        Args:
            img (PIL Image): Data image to be cropped and converted to tensor.
            gt (PIL Image): Ground truth image to be cropped and converted to tensor.

        Returns:
            Data tensor, gt tensor (tuple of tensors): cropped and converted images

        """
        x_position = coordinates[0]
        y_position = coordinates[1]

        return F.to_tensor(F.crop(img, x_position, y_position, self.crop_size, self.crop_size)), \
               F.to_tensor(F.crop(gt, x_position, y_position, self.crop_size, self.crop_size))


class CroppedDatasetGenerator:
    def __init__(self, input_path: Path, output_path, crop_size_train, crop_size_val, crop_size_test, overlap=0.5,
                 leading_zeros_length=4, override_existing=False):
        # Init list
        self.input_path = input_path
        self.output_path = output_path
        self.crop_size_train = crop_size_train
        self.crop_size_val = crop_size_val
        self.crop_size_test = crop_size_test
        self.overlap = overlap
        self.leading_zeros_length = leading_zeros_length

        self.override_existing = override_existing

        self.generator_train = CropGenerator(input_path=input_path / 'train',
                                             output_path=output_path / 'train',
                                             crop_size=crop_size_train,
                                             overlap=overlap,
                                             leading_zeros_length=leading_zeros_length,
                                             override_existing=override_existing,
                                             progress_title='Cropping "train"')
        self.generator_val = CropGenerator(input_path=input_path / 'val',
                                           output_path=output_path / 'val',
                                           crop_size=crop_size_val,
                                           overlap=overlap,
                                           leading_zeros_length=leading_zeros_length,
                                           override_existing=override_existing,
                                           progress_title='Cropping "val"')
        self.generator_test = CropGenerator(input_path=input_path / 'test',
                                            output_path=output_path / 'test',
                                            crop_size=crop_size_test,
                                            overlap=overlap,
                                            leading_zeros_length=leading_zeros_length,
                                            override_existing=override_existing,
                                            progress_title='Cropping "test"')

    def write_crops(self):
        info_list = ['Running CroppedDatasetGenerator.write_crops():',
                     f'- start_time:       \t{datetime.now():%Y-%m-%d_%H-%M-%S}',
                     f'- input_path:       \t{self.input_path}',
                     f'- output_path:      \t{self.output_path}',
                     f'- crop_size_train:  \t{self.crop_size_train}',
                     f'- crop_size_val:    \t{self.crop_size_val}',
                     f'- crop_size_test:   \t{self.crop_size_test}',
                     f'- overlap:          \t{self.overlap}',
                     f'- leading_zeros_len:\t{self.leading_zeros_length}',
                     f'- override_existing:\t{self.override_existing}',
                     '']  # empty string to get linebreak at the end when using join

        info_str = '\n'.join(info_list)
        print(info_str)

        # Write info_cropped_dataset.txt
        self.output_path.mkdir(parents=True, exist_ok=True)
        info_file = self.output_path / 'info_cropped_dataset.txt'
        with info_file.open('a') as f:
            f.write(info_str)

        print(f'Start cropping:')
        self.generator_train.write_crops()
        self.generator_val.write_crops()
        self.generator_test.write_crops()

        with info_file.open('a') as f:
            f.write(f'- end_time:         \t{datetime.now():%Y-%m-%d_%H-%M-%S}\n\n')


class CropGenerator:
    def __init__(self, input_path, output_path, crop_size, overlap=0.5, leading_zeros_length=4,
                 override_existing=False, progress_title=''):
        # Init list
        self.input_path = input_path
        self.output_path = output_path
        self.crop_size = crop_size
        self.overlap = overlap
        self.leading_zeros_length = leading_zeros_length
        self.override_existing = override_existing
        self.progress_title = progress_title

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths = get_gt_data_paths(input_path)
        self.num_imgs_in_set = len(self.img_paths)
        if self.num_imgs_in_set == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                input_path, ",".join(IMG_EXTENSIONS)))

        self.current_img_index = -1

        self.img_names_sizes, self.num_horiz_crops, self.num_vert_crops = self._get_img_size_and_crop_numbers()
        self.crop_list = self._get_crop_list()

    def write_crops(self):
        crop_function = ToTensorSlidingWindowCrop(self.crop_size)

        for img_index, x, y in tqdm(self.crop_list, desc=self.progress_title):
            self._load_image_and_var(img_index=img_index)
            coordinates = (x, y)

            img_full_name = self.img_names_sizes[img_index][0]
            img_full_name = Path(img_full_name)
            img_name = img_full_name.stem
            dest_folder_data = self.output_path / 'data' / img_name
            dest_folder_gt = self.output_path / 'gt' / img_name
            dest_folder_data.mkdir(parents=True, exist_ok=True)
            dest_folder_gt.mkdir(parents=True, exist_ok=True)

            extension = img_full_name.suffix
            filename = f'{img_name}_x{x:0{self.leading_zeros_length}d}_y{y:0{self.leading_zeros_length}d}{extension}'

            dest_filename_data = dest_folder_data / filename
            dest_filename_gt = dest_folder_gt / filename

            if not self.override_existing:
                if dest_filename_data.exists() and dest_filename_gt.exists():
                    continue

            img, gt = self.get_crops(self.current_data_img, self.current_gt_img,
                                     coordinates=coordinates, crop_function=crop_function)

            save_image(img, dest_filename_data)
            save_image(gt, dest_filename_gt)

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

    def get_crops(self, img, gt, coordinates, crop_function):
        img, gt = crop_function(img, gt, coordinates)
        return img, gt

    def _get_img_size_and_crop_numbers(self):
        # TODO documentation
        img_names_sizes = []  # list of tuples -> (gt_img_name, img_size (H, W))
        num_horiz_crops = []
        num_vert_crops = []

        for img_path, gt_path in self.img_paths:
            data_img = pil_loader(img_path)
            gt_img = pil_loader(gt_path)
            # Ensure that data and gt image are of the same size
            assert gt_img.size == data_img.size
            img_names_sizes.append((gt_path.name, data_img.size[::-1]))
            step_size = self.crop_size * self.overlap
            num_horiz_crops.append(math.ceil((data_img.size[1] - self.crop_size) / step_size + 1))
            num_vert_crops.append(math.ceil((data_img.size[0] - self.crop_size) / step_size + 1))

        return img_names_sizes, num_horiz_crops, num_vert_crops

    def _get_crop_list(self):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path',
                        help='Path to the root folder of the dataset (contains train/val/test)',
                        type=Path,
                        required=True)
    parser.add_argument('-o', '--output_path',
                        help='Path to the output folder',
                        type=Path,
                        required=True)
    parser.add_argument('-tr', '--crop_size_train',
                        help='Size of the crops in the training set',
                        type=int,
                        required=True)
    parser.add_argument('-v', '--crop_size_val',
                        help='Size of the crops in the validation set',
                        type=int,
                        required=True)
    parser.add_argument('-te', '--crop_size_test',
                        help='Size of the crops in the test set',
                        type=int,
                        required=True)
    parser.add_argument('-ov', '--overlap',
                        help='Overlap of the different crops (between 0-1)',
                        type=float,
                        default=0.5)
    parser.add_argument('-l', '--leading_zeros_length',
                        help='amount of leading zeros to encode the coordinates',
                        type=int,
                        default=4)
    parser.add_argument('-oe', '--override_existing',
                        help='If true overrides the images ',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    dataset_generator = CroppedDatasetGenerator(**args.__dict__)
    dataset_generator.write_crops()

    # example call arguments
    # -i
    # /Users/voegtlil/Documents/04_Datasets/003-DataSet/CB55-10-segmentation
    # -o
    # /Users/voegtlil/Desktop/fun
    # -tr
    # 300
    # -v
    # 300
    # -te
    # 256

    # dataset_generator = CroppedDatasetGenerator(
    #     input_path=Path('/dataset/DIVA-HisDB/segmentation/CB55'),
    #     output_path=Path('/data/usl_experiments/semantic_segmentation/datasets_cropped/CB55'),
    #     crop_size_train=300,
    #     crop_size_val=300,
    #     crop_size_test=256,
    #     overlap=0.5,
    #     leading_zeros_length=4,
    #     override_existing=False)
    # dataset_generator.write_crops()
