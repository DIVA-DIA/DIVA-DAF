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

