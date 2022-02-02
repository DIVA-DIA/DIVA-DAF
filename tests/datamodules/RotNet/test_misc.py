from src.datamodules.RotNet.utils.misc import validate_path_for_self_supervised
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

DATA_FOLDER_NAME = 'data'


def test_validate_path_for_self_supervised(data_dir_cropped):
    assert data_dir_cropped == validate_path_for_self_supervised(data_dir=data_dir_cropped,
                                                                 data_folder_name=DATA_FOLDER_NAME)