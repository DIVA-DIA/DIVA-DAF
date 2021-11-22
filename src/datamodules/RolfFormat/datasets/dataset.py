"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import torch.utils.data as data
from torch import is_tensor
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.utils.misc import ImageDimensions
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif')

log = utils.get_logger(__name__)


@dataclass
class DatasetSpecs:
    data_root: str
    doc_dir: str
    doc_names: str
    gt_dir: str
    gt_names: str
    range_from: int
    range_to: int


class DatasetRolfFormat(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png
    """

    def __init__(self, dataset_specs: List[DatasetSpecs], image_dims: ImageDimensions,
                 is_test=False, image_transform=None, target_transform=None, twin_transform=None,
                 classes=None, **kwargs):
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

        self.dataset_specs = dataset_specs

        self.image_dims = image_dims

        # Init list
        self.classes = classes

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths_per_page = self.get_gt_data_paths(list_specs=self.dataset_specs)

        self.num_samples = len(self.img_paths_per_page)

        assert self.num_samples > 0

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
        img, gt = self._apply_transformation(data_img, gt_img)
        return img, gt

    def _get_test_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        return img, gt, index

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.img_paths_per_page[index][0])
        gt_img = pil_loader(self.img_paths_per_page[index][1])

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width
        assert gt_img.height == self.image_dims.height and gt_img.width == self.image_dims.width

        return data_img, gt_img

    def _apply_transformation(self, img, gt):
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

        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt

    @staticmethod
    def _get_paths_from_specs(data_root: str,
                              doc_dir: str, doc_names: str,
                              gt_dir: str, gt_names: str,
                              range_from: int, range_to: int):

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
                paths.append((path_doc_file, path_gt_file, path_doc_file.stem))

        assert len(paths) > 0

        return paths

    @staticmethod
    def get_gt_data_paths(list_specs: List[DatasetSpecs]) -> List[Tuple[Path, Path, str]]:
        paths = []

        for specs in list_specs:
            paths += DatasetRolfFormat._get_paths_from_specs(**asdict(specs))

        return paths
