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

from torchvision.datasets.folder import has_file_allowed_extension, pil_loader
from torchvision.transforms import functional as F
from tqdm import tqdm

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif')
JPG_EXTENSIONS = ('.jpg', '.jpeg')


def get_img_paths_uncropped(directory):
    """
    Parameters
    ----------
    directory: string
        parent directory with images inside

    Returns
    -------
    paths: list of paths

    """

    paths = []
    directory = Path(directory).expanduser()

    if not directory.is_dir():
        logging.error(f'Directory not found ({directory})')

    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue

        for img_name in sorted(subdir.iterdir()):
            if has_file_allowed_extension(str(img_name), IMG_EXTENSIONS):
                paths.append((subdir / img_name, str(subdir.stem)))

    return paths


class ImageCrop(object):
    """
    Crop the data and ground truth image at the specified coordinates to the specified size and convert
    them to a tensor.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, coordinates):
        """
        Args:
            img (PIL Image): Data image to be cropped and converted to tensor.
            gt (PIL Image): Ground truth image to be cropped and converted to tensor.

        Returns:
            Data tensor, gt tensor (tuple of tensors): cropped and converted images

        """
        x_position = coordinates[0]
        y_position = coordinates[1]

        img_crop = F.to_tensor(
            F.crop(img=img, left=x_position, top=y_position, width=self.crop_size, height=self.crop_size))

        return img_crop


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
                     f'- full_command:',
                     f'python tools/generate_cropped_dataset.py -i {self.input_path} -o {self.output_path} '
                     f'-tr {self.crop_size_train} -v {self.crop_size_val} -te {self.crop_size_test} -ov {self.overlap} '
                     f'-l {self.leading_zeros_length}',
                     f'',
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

        self.step_size = int(self.crop_size * (1 - self.overlap))

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths = get_img_paths_uncropped(input_path)
        self.num_imgs_in_set = len(self.img_paths)
        if self.num_imgs_in_set == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                input_path, ",".join(IMG_EXTENSIONS)))

        self.current_split = ''
        self.current_img_index = -1

        self.img_names_sizes, self.num_horiz_crops, self.num_vert_crops = self._get_img_size_and_crop_numbers()
        self.crop_list = self._get_crop_list()

    def write_crops(self):
        crop_function = ImageCrop(self.crop_size)

        for img_index, x, y in tqdm(self.crop_list, desc=self.progress_title):
            self._load_image(img_index=img_index)
            coordinates = (x, y)

            split_name = self.img_names_sizes[img_index][0]
            img_full_name = self.img_names_sizes[img_index][1]
            img_full_name = Path(img_full_name)
            img_name = img_full_name.stem
            dest_folder = self.output_path / split_name / img_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            extension = img_full_name.suffix
            filename = f'{img_name}_x{x:0{self.leading_zeros_length}d}_y{y:0{self.leading_zeros_length}d}{extension}'

            dest_filename = dest_folder / filename

            if not self.override_existing:
                if dest_filename.exists():
                    continue

            img = self.get_crop(self.current_img, coordinates=coordinates, crop_function=crop_function)

            pil_img = F.to_pil_image(img, mode='RGB')

            if extension in JPG_EXTENSIONS:
                pil_img.save(dest_filename, quality=95)
            else:
                # save_image(img, dest_filename)
                pil_img.save(dest_filename)

    def _load_image(self, img_index):
        """
        Inits the variables responsible of tracking which crop should be taken next, the current images and the like.
        This should be run every time a new page gets loaded for the test-set
        """

        if self.current_img_index == img_index:
            return

        # Load image
        self.current_img = pil_loader(self.img_paths[img_index][0])

        # Update pointer to current image
        self.current_img_index = img_index
        self.current_split = self.img_paths[img_index][1]

    def get_crop(self, img, coordinates, crop_function):
        img = crop_function(img, coordinates)
        return img

    def _get_img_size_and_crop_numbers(self):
        img_names_sizes = []  # list of tuples -> (split_name, img_name, img_size (H, W))
        num_horiz_crops = []
        num_vert_crops = []

        for img_path, split_name in self.img_paths:
            data_img = pil_loader(img_path)
            img_names_sizes.append((split_name, img_path.name, data_img.size))
            num_horiz_crops.append(math.ceil((data_img.size[0] - self.crop_size) / self.step_size + 1))
            num_vert_crops.append(math.ceil((data_img.size[1] - self.crop_size) / self.step_size + 1))

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
            x_position = self.img_names_sizes[img_index][2][0] - self.crop_size
        else:
            x_position = self.step_size * hcrop_index
            assert x_position < self.img_names_sizes[img_index][2][0] - self.crop_size

        # Y coordinate
        if vcrop_index == self.num_vert_crops[img_index] - 1:
            # We are at the bottom end
            y_position = self.img_names_sizes[img_index][2][1] - self.crop_size
        else:
            y_position = self.step_size * vcrop_index
            assert y_position < self.img_names_sizes[img_index][2][1] - self.crop_size

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

    # example call arguments
    # -i
    # /dataset/DIVA-HisDB/segmentation/CB55
    # -o
    # /net/research-hisdoc/datasets/semantic_segmentation/datasets_cropped/temp-CB55
    # -tr
    # 300
    # -v
    # 300
    # -te
    # 256

    # dataset_generator = CroppedDatasetGenerator(
    #     input_path=Path('/dataset/DIVA-HisDB/segmentation/CB55'),
    #     output_path=Path('/net/research-hisdoc/datasets/semantic_segmentation/datasets_cropped/CB55'),
    #     crop_size_train=300,
    #     crop_size_val=300,
    #     crop_size_test=256,
    #     overlap=0.5,
    #     leading_zeros_length=4,
    #     override_existing=False)
    # dataset_generator.write_crops()
