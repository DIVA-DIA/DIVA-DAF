import os
import pytest

from pathlib import Path

import shutil

from PIL import Image
from torchvision.datasets.folder import pil_loader


def _get_result_imgs(tmp_path, filename: str) -> Image:
    """
    Moves the test data into the tmp path of the testing environment.
    :param tmp_path:
    :return:
    """
    w_filename = __file__
    test_dir = Path(w_filename).parent / 'data' / filename

    if test_dir.is_file():
        shutil.copyfile(str(test_dir), str(tmp_path / filename))

    return pil_loader(str(tmp_path / filename))


@pytest.fixture
def result_img_2_3_horizontal(tmp_path):
    return _get_result_imgs(tmp_path=tmp_path, filename='2_3_horizontal.png')


@pytest.fixture
def result_img_2_3_vertical(tmp_path):
    return _get_result_imgs(tmp_path=tmp_path, filename='2_3_vertical.png')


@pytest.fixture
def result_img_2_3_horizontal_vertical(tmp_path):
    return _get_result_imgs(tmp_path=tmp_path, filename='2_3_horizontal_vertical.png')
