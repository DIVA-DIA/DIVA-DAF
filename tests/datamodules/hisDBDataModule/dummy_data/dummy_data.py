import os
from distutils import dir_util

from pytest import fixture


@fixture
def data_dir(tmp_path):
    """
    Moves the test data into the tmp path of the testing environment.
    :param tmp_path:
    :param request:
    :return:
    """
    filename = __file__
    test_dir = os.path.join(os.path.dirname(filename), 'dummy_dataset')

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmp_path))

    return tmp_path


@fixture
def data_dir_cropped(tmp_path):
    """
    Moves the test data into the tmp path of the testing environment.
    :param tmp_path:
    :return:
    """
    filename = __file__
    test_dir = os.path.join(os.path.dirname(filename), 'dummy_dataset_cropped')

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmp_path))

    return tmp_path
