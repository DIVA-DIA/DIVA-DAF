from pathlib import Path

from src.datamodules.utils.exceptions import PathMissingDirinSplitDir, PathNone, PathNotDir, PathMissingSplitDir


def validate_path_for_self_supervised(data_dir: str, data_folder_name: str) -> Path:
    """
    Validates the path for the self-supervised learning task. The path should contain a train/val/test folder
    and each of them a folder with the name of the data_folder_name.

    :param data_dir: Root dir of the dataset (folder containing the train/val/test folder)
    :type data_dir: str
    :param data_folder_name: Name of the folder containing the data
    :type data_folder_name: str
    :raises PathNone: If data_dir is None
    :raises PathNotDir: If data_dir is not a directory
    :raises PathMissingSplitDir: If data_dir does not contain train/val/test
    :raises PathMissingDirinSplitDir: If train/val/test does not contain data_folder_name
    :return: Path to the root dir of the dataset
    :rtype: Path
    """
    if data_dir is None:
        raise PathNone("Please provide the path to root dir of the dataset "
                       "(folder containing the train/val/test folder)")
    else:
        split_names = ['train', 'val', 'test']
        type_names = [data_folder_name]

        data_folder = Path(data_dir)
        if not data_folder.is_dir():
            raise PathNotDir("Please provide the path to root dir of the dataset "
                             "(folder containing the train/val/test folder)")
        split_folders = [d for d in data_folder.iterdir() if d.is_dir() and d.name in split_names]
        if len(split_folders) != 3:
            raise PathMissingSplitDir(f'Your path needs to contain train/val/test and '
                                      f'each of them a folder {data_folder_name}')

        # check if we have train/test/val
        for split in split_folders:
            type_folders = [d for d in split.iterdir() if d.is_dir() and d.name in type_names]
            # check if we have data/gt
            if len(type_folders) != 1:
                raise PathMissingDirinSplitDir(f'Folder {split.name} does not contain a {data_folder_name}')
    return Path(data_dir)
