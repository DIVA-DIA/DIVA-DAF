from typing import Union, List, Optional

from torchvision import transforms

from src.datamodules.Classification.datamodule import ClassificationDatamodule
from src.utils import utils

log = utils.get_logger(__name__)


class ClassificationDatamoduleImagenet(ClassificationDatamodule):
    """
    Datamodule for a classification task. It takes advantage of the ImageFolder class from PyTorch

    The data is expected to be in the following format::

        data_dir
            ├── train
            │   ├── 0
            │   │   ├── image_1.png
            │   │   ├── ...
            │   │   └── image_N.png
            │   ├── ...
            │   └── N
            │       ├── image_1.png
            │       ├── ...
            │       └── image_N.png
            ├──  val
            │   ├── 0
            │   │   ├── image_1.png
            │   │   ├── ...
            │   │   └── image_N.png
            │   ├── ...
            │   └── N
            │       ├── image_1.png
            │       ├── ...
            │       └── image_N.png
            └── test
                ├── 0
                │   ├── image_1.png
                │   ├── ...
                │   └── image_N.png
                ├── ...
                └── N
                    ├── image_1.png
                    ├── ...
                    └── image_N.png

    :param data_dir: Path to the root directory of the dataset.
    :type data_dir: str
    :param selection_train: Either an integer or a list of strings. If an integer is provided, the first n classes are
        selected. If a list of strings is provided, the classes with the given names are selected.
    :type selection_train: Optional[Union[int, List[str]]]
    :param selection_val: Either an integer or a list of strings. If an integer is provided, the first n classes are
        selected. If a list of strings is provided, the classes with the given names are selected.
    :type selection_val: Optional[Union[int, List[str]]]
    :param num_workers: Number of workers for the dataloaders.
    :type num_workers: int
    :param batch_size: Batch size for the dataloaders.
    :type batch_size: int
    :param shuffle: Whether to shuffle the data.
    :type shuffle: bool
    :param drop_last: Whether to drop the last batch if it is smaller than the batch size.
    :type drop_last: bool
    """

    def __init__(self, data_dir: str,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """
        Constructor method for the ClassificationDatamodule class.
        """
        super().__init__(data_dir=data_dir, selection_train=selection_train, selection_val=selection_val,
                         num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
