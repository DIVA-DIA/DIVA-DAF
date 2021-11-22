from pathlib import Path

from src.datamodules.utils.exceptions import PathMissingDirinSplitDir, PathNone, PathNotDir, PathMissingSplitDir


def validate_path_for_self_supervised(data_dir, data_folder_name: str):
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
