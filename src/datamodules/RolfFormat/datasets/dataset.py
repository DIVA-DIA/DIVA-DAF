"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import torch.utils.data as data
from torch import is_tensor
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.utils.misc import ImageDimensions, get_output_file_list
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif')

log = utils.get_logger(__name__)


@dataclass
class DatasetSpecs:
    """
    This class is used to specify the location of the data and ground truth files. It can also be used to
    specify a range of files that should be used. This is useful if you want to split the data into train/val/test
    and want to use the same data root for all three splits.
    """
    data_root: str
    doc_dir: str
    doc_names: str
    gt_dir: str
    gt_names: str
    range_from: int
    range_to: int


class DatasetRolfFormat(data.Dataset):
    """A generic data loader where the images are arranged in this way::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png

        :param dataset_specs: The dataset specs that specify the location of the data and ground truth files.
        :type dataset_specs: List[DatasetSpecs]
        :param image_dims: The dimensions of the images.
        :type image_dims: ImageDimensions
        :param is_test: Is it the test dataset?
        :type is_test: bool
        :param image_transform: Transformations that should be applied to the image.
        :type image_transform: callable
        :param target_transform: Transformations that should be applied to the ground truth.
        :type target_transform: callable
        :param twin_transform: Transformations that should be applied to both the image and the ground truth.
        :type twin_transform: callable
    """

    def __init__(self, dataset_specs: List[DatasetSpecs], image_dims: ImageDimensions,
                 is_test: bool = False, image_transform: callable = None, target_transform: callable = None,
                 twin_transform: callable = None):
        """
        Constructor method for the DatasetRolfFormat class.
        """

        self.dataset_specs = dataset_specs

        self.image_dims = image_dims

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_gt_path_list = self.get_img_gt_path_list(list_specs=self.dataset_specs)

        if is_test:
            self.image_path_list = [img_gt_path[0] for img_gt_path in self.img_gt_path_list]
            self.output_file_list = get_output_file_list(image_path_list=self.image_path_list)

        self.num_samples = len(self.img_gt_path_list)

        assert self.num_samples > 0

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Union[Tuple[Image.Image, Image.Image], Tuple[Image.Image, Image.Image, int]]:
        """
        This function returns the image and the ground truth for a given index. If it is the test dataset,

        :param index: The index of the sample that should be returned.
        :type index: int
        :return: The image and the ground truth for the given index.
        :rtype: tuple
        """
        if self.is_test:
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        This function returns the image and the ground truth for a given index.

        :param index: The index of the sample that should be returned.
        :type index: int
        :return: The image and the ground truth for the given index.
        :rtype: tuple
        """
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        return img, gt

    def _get_test_items(self, index: int) -> Tuple[Image.Image, Image.Image, int]:
        """
        This function returns the image and the ground truth for a given index.

        :param index: The index of the sample that should be returned.
        :type index: int
        :return: The image and the ground truth for the given index with the index.
        :rtype: tuple
        :return:
        """
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        return img, gt, index

    def _load_data_and_gt(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        This function loads the image and the ground truth for a given index.

        :param index: The index of the sample that should be returned.
        :type index: int
        :return: The image and the ground truth for the given index.
        :rtype: tuple
        """
        data_img = pil_loader(str(self.img_gt_path_list[index][0]))
        gt_img = pil_loader(str(self.img_gt_path_list[index][1]))

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width
        assert gt_img.height == self.image_dims.height and gt_img.width == self.image_dims.width

        return data_img, gt_img

    def _apply_transformation(self, img: Image.Image, gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param img: The original image onto which the transformations should be applied.
        :type img: Image.Image
        :param gt: The ground truth onto which the transformations should be applied.
        :type gt: Image.Image
        :return: The transformed image and ground truth.
        :rtype: Tuple[Image.Image, Image.Image]
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

        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt

    @staticmethod
    def _get_paths_from_specs(data_root: str,
                              doc_dir: str, doc_names: str,
                              gt_dir: str, gt_names: str,
                              range_from: int, range_to: int) -> List[Tuple[Path, Path]]:
        """
        This function returns a list of tuples that contain the path to the gt and image that belong together.

        :param data_root: The root where the data is located.
        :type data_root: str
        :param doc_dir: The directory where the images are located.
        :type doc_dir: str
        :param doc_names: The name of the images.
        :type doc_names: str
        :param gt_dir: The directory where the ground truth is located.
        :type gt_dir: str
        :param gt_names: The name of the ground truth.
        :type gt_names: str
        :param range_from: The first index of the range that should be used.
        :type range_from: int
        :param range_to: The last index of the range that should be used.
        :type range_to: int
        :return: A list of tuples that contain the path to the gt and image that belong together.
        :rtype: List[Tuple[Path, Path]]
        """

        path_root = Path(data_root)
        path_doc_dir = path_root / doc_dir
        path_gt_dir = path_root / gt_dir

        if not path_doc_dir.is_dir():
            log.error(f'Document directory not found ("{path_doc_dir}")!')

        if not path_gt_dir.is_dir():
            log.error(f'Ground Truth directory not found ("{path_gt_dir}")!')

        p = re.compile('#+')

        # assert that there is exactly one placeholder group
        assert len(p.findall(doc_names)) == 1
        assert len(p.findall(gt_names)) == 1

        search_doc_names = p.search(doc_names)
        doc_prefix = doc_names[:search_doc_names.span(0)[0]]
        doc_suffix = doc_names[search_doc_names.span(0)[1]:]
        doc_number_length = len(search_doc_names.group(0))

        search_gt_names = p.search(gt_names)
        gt_prefix = gt_names[:search_gt_names.span(0)[0]]
        gt_suffix = gt_names[search_gt_names.span(0)[1]:]
        gt_number_length = len(search_gt_names.group(0))

        paths = []
        for i in range(range_from, range_to + 1):
            doc_filename = f'{doc_prefix}{i:0{doc_number_length}d}{doc_suffix}'
            path_doc_file = path_doc_dir / doc_filename

            gt_filename = f'{gt_prefix}{i:0{gt_number_length}d}{gt_suffix}'
            path_gt_file = path_gt_dir / gt_filename

            assert path_doc_file.exists() == path_gt_file.exists()

            if path_doc_file.exists() and path_gt_file.exists():
                paths.append((path_doc_file, path_gt_file))

        assert len(paths) > 0

        return paths

    @staticmethod
    def get_img_gt_path_list(list_specs: List[DatasetSpecs]) -> List[Tuple[Path, Path]]:
        """
        Returns a list of tuples that contain the path to the gt and image that belong together.

        :param list_specs: The dataset specs that specify the location of the data and ground truth files.
        :type list_specs: List[DatasetSpecs]
        :return: A list of tuples that contain the path to the gt and image that belong together.
        :rtype: List[Tuple[Path, Path]]
        """
        paths = []

        for specs in list_specs:
            paths += DatasetRolfFormat._get_paths_from_specs(**asdict(specs))

        return paths
