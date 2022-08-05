import pytest
from src.datamodules.base_datamodule import AbstractDatamodule


def test_check_min_num_samples():
    AbstractDatamodule.check_min_num_samples(num_devices=2, batch_size1=5, num_samples=10, data_split='train',
                                             drop_last=True)


def test_check_min_num_samples_drop_error():
    with pytest.raises(ValueError):
        AbstractDatamodule.check_min_num_samples(num_devices=2, batch_size1=20, num_samples=10, data_split='train',
                                                 drop_last=True)


def test_check_min_num_samples_no_drop_error(caplog):
    num_samples = 10
    data_split = 'train'
    batch_size = 20
    AbstractDatamodule.check_min_num_samples(num_devices=4, batch_size1=batch_size, num_samples=num_samples,
                                             data_split=data_split,
                                             drop_last=False)
    assert f'WARNING  src.datamodules.base_datamodule:rank_zero.py:32 ' \
           f'#samples ({num_samples}) in "{data_split}" smaller than ' \
           f'#devices (4) times batch size ({batch_size}). ' \
           f'This works due to drop_last=False, however samples might occur multiple times. ' \
           f'Check if this behavior is intended!\n' in caplog.text
