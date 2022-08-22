"""
Load a dataset of historic documents by specifying the folder where its located.
"""

import argparse
# Utils
import itertools
import json
import logging
import math
import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets.folder import has_file_allowed_extension, pil_loader
from tqdm import tqdm
from typing import Tuple, List

from src.datamodules.utils.misc import ImageDimensions

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif')


def get_img_paths(directory):
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

    for img_path in sorted(directory.iterdir()):
        if has_file_allowed_extension(str(img_path), IMG_EXTENSIONS):
            paths.append(img_path)

    return paths


class TiledDatasetGenerator:
    def __init__(self, input_path: Path, output_path: Path, rows: int, cols: int, override_existing=False):
        # Init list
        self.input_path = input_path
        self.output_path = output_path
        self.rows = rows
        self.cols = cols

        self.override_existing = override_existing

        self.permutations = self._get_permutations()

        self.generator = TileGenerator(input_path=input_path,
                                       output_path=output_path,
                                       rows=rows,
                                       cols=cols,
                                       permutations=self.permutations,
                                       override_existing=override_existing,
                                       progress_title='Creating tiles')

    def write_tiles(self):
        info_list = ['Running TiledDatasetGenerator.write_tiles():',
                     f'- full_command:',
                     f'python tools/generate_tiles_dataset.py -i {self.input_path} -o {self.output_path} '
                     f'-r {self.rows} -c {self.cols}'
                     f'',
                     f'- start_time:       \t{datetime.now():%Y-%m-%d_%H-%M-%S}',
                     f'- input_path:       \t{self.input_path}',
                     f'- output_path:      \t{self.output_path}',
                     f'- rows:  \t{self.rows}',
                     f'- cols:    \t{self.cols}',
                     f'- override_existing:\t{self.override_existing}',
                     '']  # empty string to get linebreak at the end when using join

        info_str = '\n'.join(info_list)
        print(info_str)

        permutation_file = self.output_path / 'permutations.json'
        with open(permutation_file, 'w') as f:
            json.dump(self.permutations, f)

        # Write info_cropped_dataset.txt
        self.output_path.mkdir(parents=True, exist_ok=True)
        info_file = self.output_path / 'info_permutation_dataset.txt'
        with info_file.open('a') as f:
            f.write(info_str)

        print(f'Start tiling:')
        self.generator.write_tiles()

        with info_file.open('a') as f:
            f.write(f'- end_time:         \t{datetime.now():%Y-%m-%d_%H-%M-%S}\n\n')
            f.write(f'- permutations:     \tindex: \tpermutation: \n')
            for i, perm in enumerate(self.permutations):
                f.write(f'\t\t{i} \t{perm}\n')

    def _get_permutations(self) -> List[Tuple]:
        # # Get permutations and col permutations
        # row_perms = list(itertools.product(range(math.factorial(self.cols)), repeat=self.rows))
        # # row_perms
        # cols_perms = list(itertools.permutations(range(self.cols)))
        # return row_perms, cols_perms
        return sorted(list(itertools.permutations(range(self.rows * self.cols))))


class TileGenerator:
    def __init__(self, input_path, output_path, rows, cols, permutations,
                 override_existing=False, progress_title=''):
        # Init list
        self.input_path = input_path
        self.output_path = output_path
        self.rows = rows
        self.cols = cols
        self.permutations = permutations
        self.override_existing = override_existing
        self.progress_title = progress_title

        # List paths to the images
        self.img_paths = get_img_paths(input_path)
        self.num_imgs_in_set = len(self.img_paths)
        if self.num_imgs_in_set == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                input_path, ",".join(IMG_EXTENSIONS)))

        self.current_img_index = -1

    def write_tiles(self):
        for img_index in tqdm(range(self.num_imgs_in_set), desc=self.progress_title):
            self._load_image_and_center_crop(img_index=img_index)

            if self.center_cropped_image.width % self.cols or self.center_cropped_image.height % self.rows:
                raise RuntimeError(
                    f"Image dimensions({self.center_cropped_image.width, self.center_cropped_image.height}) "
                    f"are not divisible by the number of rows and columns")
            img_name = self.img_paths[img_index].name.replace('_max', '')
            tile_dims = ImageDimensions(width=self.center_cropped_image.width // self.cols,
                                        height=self.center_cropped_image.height // self.rows)

            pool = multiprocessing.Pool(multiprocessing.cpu_count())

            parameters = zip(itertools.repeat(self.output_path), itertools.repeat(img_name),
                             itertools.repeat(self.current_img), itertools.repeat(self.center_cropped_image),
                             self.permutations, range(len(self.permutations)), itertools.repeat(tile_dims),
                             itertools.repeat(self.override_existing),
                             itertools.repeat(self.rows), itertools.repeat(self.cols))

            pool.starmap(func=self.create_image_by_permutation, iterable=parameters)
            # for each permutation

    @staticmethod
    def create_image_by_permutation(output_path, img_name, current_img, center_cropped_image, permutation,
                                    permutation_id, tile_dims, override_existing, rows, cols):
        dest_folder = output_path / str(permutation_id)
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_filename = dest_folder / img_name

        if not override_existing:
            if dest_filename.exists():
                return

        img_tiled_array = TileGenerator.tile_image(
            current_img=current_img, center_cropped_image=center_cropped_image, permutation=permutation,
            tile_dims=tile_dims, rows=rows, cols=cols)

        pil_img_tiled = Image.fromarray(img_tiled_array.astype(np.uint8))
        pil_img_tiled.save(dest_filename)

    def _load_image_and_center_crop(self, img_index):
        """
        Loads the next image based on the img_index and crops the center of the image
        """

        if self.current_img_index == img_index:
            return

        # Load image
        self.current_img = pil_loader(self.img_paths[img_index])

        # Center crop image 866 x 1236 should be the size at the end. We need an offset on all sides of 3 pixels
        # to get during training a size of 860 x 1230
        self.center_cropped_image = self._center_crop(width=860, height=1230)

        # Update pointer to current image
        self.current_img_index = img_index

    def _center_crop(self, width, height):
        left = (self.current_img.width - width) // 2
        top = (self.current_img.height - height) // 2
        right = left + width
        bottom = top + height
        return self.current_img.crop((left, top, right, bottom))

    @staticmethod
    def tile_image(current_img, center_cropped_image, permutation, tile_dims, rows, cols):
        cropped_img_array = np.array(center_cropped_image)
        current_img_array = np.array(current_img)
        permutation = np.array(permutation).reshape((rows, cols))
        new_img_array = current_img_array.copy()
        for i in range(rows):
            for j in range(cols):
                width_offset = ((current_img_array.shape[1] - cropped_img_array.shape[1]) // 2)
                height_offset = ((current_img_array.shape[0] - cropped_img_array.shape[0]) // 2)

                width_start_cropped = ((permutation[i, j] % cols) * tile_dims.width)
                width_end_cropped = width_start_cropped + tile_dims.width
                height_start_cropped = ((permutation[i, j] // cols) * tile_dims.height)
                height_end_cropped = height_start_cropped + tile_dims.height

                width_start_ori = width_offset + (j * tile_dims.width)
                width_end_ori = width_offset + ((j + 1) * tile_dims.width)
                height_start_ori = height_offset + (i * tile_dims.height)
                height_end_ori = height_offset + ((i + 1) * tile_dims.height)

                new_img_array[height_start_ori: height_end_ori, width_start_ori: width_end_ori, :] = cropped_img_array[
                                                                                                     height_start_cropped:height_end_cropped,
                                                                                                     width_start_cropped:width_end_cropped]

        return new_img_array


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
    parser.add_argument('-r', '--rows',
                        help='Number of rows in the tiled image',
                        type=int,
                        required=True)
    parser.add_argument('-c', '--cols',
                        help='Number of columns in the tiled image',
                        type=int,
                        required=True)
    parser.add_argument('-oe', '--override_existing',
                        help='If true overrides the images ',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    dataset_generator = TiledDatasetGenerator(**args.__dict__)
    dataset_generator.write_tiles()
