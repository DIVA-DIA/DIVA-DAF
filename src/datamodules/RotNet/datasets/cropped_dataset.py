"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
from pathlib import Path
from typing import List, Union, Optional, Tuple

from omegaconf import ListConfig
from PIL import Image
from torch import is_tensor, Tensor
from torchvision.datasets.folder import has_file_allowed_extension, pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.DivaHisDB.datasets.cropped_dataset import CroppedHisDBDataset
from src.datamodules.utils.misc import selection_validation
from src.datamodules.utils.single_transforms import RightAngleRotation
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm')

log = utils.get_logger(__name__)


class CroppedRotNet(CroppedHisDBDataset):
    """
    Dataset implementation of the RotNet paper of `Gidaris et al. <https://arxiv.org/abs/1803.07728>`_. This
    dataset is used for the DivaHisDB dataset in a cropped setup.

    The structure of the folder should be as follows::

         data_dir
        ├── train_folder_name
        │   ├── data_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image_crop_1.png
        │   │   │   ├── ...
        │   │   │   └── image_crop_N.png
        │   │   └──original_image_name_N
        │   │       ├── image_crop_1.png
        │   │       ├── ...
        │   │       └── image_crop_N.png
        │   └── gt_folder_name
        │       ├── original_image_name_1
        │       │   ├── image_crop_1.png
        │       │   ├── ...
        │       │   └── image_crop_N.png
        │       └──original_image_name_N
        │           ├── image_crop_1.png
        │           ├── ...
        │           └── image_crop_N.png
        ├── validation_folder_name
        │   ├── data_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image_crop_1.png
        │   │   │   ├── ...
        │   │   │   └── image_crop_N.png
        │   │   └──original_image_name_N
        │   │       ├── image_crop_1.png
        │   │       ├── ...
        │   │       └── image_crop_N.png
        │   └── gt_folder_name
        │       ├── original_image_name_1
        │       │   ├── image_crop_1.png
        │       │   ├── ...
        │       │   └── image_crop_N.png
        │       └──original_image_name_N
        │           ├── image_crop_1.png
        │           ├── ...
        │           └── image_crop_N.png
        └── test_folder_name
            ├── data_folder_name
            │   ├── original_image_name_1
            │   │   ├── image_crop_1.png
            │   │   ├── ...
            │   │   └── image_crop_N.png
            │   └──original_image_name_N
            │       ├── image_crop_1.png
            │       ├── ...
            │       └── image_crop_N.png
            └── gt_folder_name
                ├── original_image_name_1
                │   ├── image_crop_1.png
                │   ├── ...
                │   └── image_crop_N.png
                └──original_image_name_N
                    ├── image_crop_1.png
                    ├── ...
                    └── image_crop_N.png

    :param path: Path to root dir of the dataset (folder containing the train/val/test folder)
    :type path: Path
    :param data_folder_name: Name of the folder containing the train/val/test folder
    :type data_folder_name: str
    :param gt_folder_name: Name of the folder containing the train/val/test folder
    :type gt_folder_name: str
    :param selection: If you only want to use a subset of the dataset, you can specify the name of the files
        (without the file extension) in a list. If you want to use all files, set this parameter to None.
    :type selection: Union[int, List[str]]
    :param is_test: If True, the it returns additional information that are important for the test set.
    :type is_test: bool
    :param image_transform:
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str = None,
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test: bool = False, image_transform: callable = None):
        """
        Constructor method of the class RotNetDataset.
        """
        super(CroppedRotNet, self).__init__(path=path, data_folder_name=data_folder_name, gt_folder_name=gt_folder_name,
                                            selection=selection,
                                            is_test=is_test, image_transform=image_transform,
                                            target_transform=None, twin_transform=None)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        This function returns the image and the ground truth for a given index.

        :param index: index of the image
        :type index: int
        :return: the image and the ground truth
        :rtype: Tuple[Tensor, int]
        """
        data_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, index=index)
        return img, gt

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def _load_data_and_gt(self, index: int) -> Image.Image:
        """
        Loads the image for a given index.

        :param index: index of the image to be loaded
        :type index: int
        :return: the image
        :rtype: Image.Image
        """

        data_img = pil_loader(self.img_paths_per_page[index])
        return data_img

    def _apply_transformation(self, img: Image.Image, index: int) -> Tuple[Tensor, int]:
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param img: PIL image of the codex
        :type img: Image.Image
        :param index: index of the image to determine the rotation angle
        :type index: int
        """
        if self.twin_transform is not None and not self.is_test:
            img, _ = self.twin_transform(img, None)

        if self.image_transform is not None:
            # perform transformations
            img, _ = self.image_transform(img, None)

        if not is_tensor(img):
            img = ToTensor()(img)

        rotation_transformation = RightAngleRotation()
        img = rotation_transformation(img)

        return img, rotation_transformation.target_class

    @staticmethod
    def get_gt_data_paths(directory: Path, data_folder_name: str, gt_folder_name: str = None,
                          selection: Optional[Union[int, List[str]]] = None) \
            -> List[Path]:
        """
        Creates the list of paths to the original images.

        Structure of the folder::

            dictionary
            ├── data_folder_name
            │   ├── original_image_name_1
            │   │   ├── image_crop_1.png
            │   │   ├── ...
            │   │   └── image_crop_N.png
            │   └──original_image_name_N
            │       ├── image_crop_1.png
            │       ├── ...
            │       └── image_crop_N.png
            └── gt_folder_name
                ├── original_image_name_1
                │   ├── image_crop_1.png
                │   ├── ...
                │   └── image_crop_N.png
                └──original_image_name_N
                    ├── image_crop_1.png
                    ├── ...
                    └── image_crop_N.png

        :param directory: Path to root dir of split
        :type directory: Path
        :param data_folder_name: Name of the folder containing the data
        :type data_folder_name: str
        :param gt_folder_name: Name of the folder containing the ground truth
        :type gt_folder_name: str
        :param selection: If you only want to use a subset of the dataset, you can specify the name of the files
            (without the file extension) in a list. If you want to use all files, set this parameter to None.
        :type selection: Union[int, List[str]]
        :return: List of paths to the original images
        :rtype: List[Path]
        """
        paths = []
        directory = directory.expanduser()

        path_data_root = directory / data_folder_name

        if not (path_data_root.is_dir()):
            log.error("folder data or gt not found in " + str(directory))

        # get all subitems (and files) sorted
        subitems = sorted(path_data_root.iterdir())

        # check the selection parameter
        if selection:
            selection = selection_validation(subitems, selection, full_page=False)

        counter = 0  # Counter for subdirectories, needed for selection parameter

        for path_data_subdir in subitems:
            if not path_data_subdir.is_dir():
                if has_file_allowed_extension(path_data_subdir.name, IMG_EXTENSIONS):
                    log.warning("image file found in data root: " + str(path_data_subdir))
                continue

            counter += 1

            if selection:
                if isinstance(selection, int):
                    if counter > selection:
                        break

                elif isinstance(selection, ListConfig) or isinstance(selection, list):
                    if path_data_subdir.name not in selection:
                        continue

            for path_data_file in sorted(path_data_subdir.iterdir()):
                if has_file_allowed_extension(path_data_file.name, IMG_EXTENSIONS):
                    paths.append(path_data_file)

        return paths
