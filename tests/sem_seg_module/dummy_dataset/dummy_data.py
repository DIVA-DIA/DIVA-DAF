import os
from distutils import dir_util
from pathlib import Path

from pytest import fixture


@fixture
def data_dir(tmp_path):
    """
    Moves the test data into the tmp path of the testing environment.
    :param tmp_path:
    :return:
    """
    test_dir = Path(__file__).parent

    assert test_dir.is_dir(), f"{str(test_dir)} is not a directory!"
    dir_util.copy_tree(test_dir, str(tmp_path))

    return tmp_path
