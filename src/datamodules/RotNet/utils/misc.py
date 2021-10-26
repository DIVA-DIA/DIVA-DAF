"""
General purpose utility functions.

"""

from pathlib import Path

# Utils
import numpy as np
from PIL import Image

from src.datamodules.utils.exceptions import PathMissingDirinSplitDir, PathNone, PathNotDir, PathMissingSplitDir

try:
    import accimage
except ImportError:
    accimage = None


def has_extension(filename, extensions):
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


def validate_path_for_self_supervised(data_dir, data_folder_name: str = 'data'):
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
