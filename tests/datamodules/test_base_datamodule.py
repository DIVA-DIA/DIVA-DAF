import pytest
import re
from src.datamodules.base_datamodule import AbstractDatamodule


def test_check_min_num_samples():
    AbstractDatamodule.check_min_num_samples(num_devices=2, batch_size_input=5, num_samples=10, data_split='train',
                                             drop_last=True)


def test_check_min_num_samples_drop_error():
    with pytest.raises(ValueError):
        AbstractDatamodule.check_min_num_samples(num_devices=2, batch_size_input=20, num_samples=10, data_split='train',
                                                 drop_last=True)


def test_check_min_num_samples_no_drop_error(caplog):
    num_samples = 10
    data_split = 'train'
    batch_size = 20
    AbstractDatamodule.check_min_num_samples(num_devices=4, batch_size_input=batch_size, num_samples=num_samples,
                                             data_split=data_split,
                                             drop_last=False)
    expect_base = fr'WARNING  src\.datamodules\.base_datamodule:rank_zero.py:\d\d '
    expect_1 = f'#samples ({num_samples}) in "{data_split}" smaller than '
    expect_2 = f'#devices (4) times batch size ({batch_size}). '
    expect_3 = f'This works due to drop_last=False, however samples might occur multiple times. '
    expect_4 = f'Check if this behavior is intended!\n'
    assert re.search(expect_base, caplog.text)
    assert expect_1 in caplog.text
    assert expect_2 in caplog.text
    assert expect_3 in caplog.text
    assert expect_4 in caplog.text
