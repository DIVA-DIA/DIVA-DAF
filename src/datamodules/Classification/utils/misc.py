from pathlib import Path

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir


def validate_path_for_classification(data_dir: str) -> Path:
    """
    Checks if the path is valid for classification

    :param data_dir: path to the root dir of the dataset
    :type data_dir: str
    :return: path to the root dir of the dataset
    :rtype: Path
    """
    if data_dir is None:
        raise PathNone("Please provide the path to root dir of the dataset "
                       "(folder containing the train/val folder)")
    else:
        split_names = ['train', 'val']

        data_folder = Path(data_dir)
        if not data_folder.is_dir():
            raise PathNotDir("Please provide the path to root dir of the dataset "
                             "(folder containing the train/val folder)")
        split_folders = [d for d in data_folder.iterdir() if d.is_dir() and d.name in split_names]
        if len(split_folders) != 2:
            raise PathMissingSplitDir('Your path needs to contain train/val and '
                                      'each of them a folder per class')

    return Path(data_dir)
