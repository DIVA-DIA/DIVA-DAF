import pytest

from src.datamodules.RolfFormat.datasets.dataset import DatasetRolfFormat, DatasetSpecs
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_rolf.dummy_data import data_dir


@pytest.fixture
def dataset_test(data_dir):
    dataset_specs = [_get_dataspecs(data_dir, train=False)]
    return DatasetRolfFormat(dataset_specs=dataset_specs, image_dims=ImageDimensions(width=960, height=1344),
                             is_test=True)


@pytest.fixture
def dataset_train(data_dir):
    dataset_specs = [_get_dataspecs(data_dir, train=True)]
    return DatasetRolfFormat(dataset_specs=dataset_specs, image_dims=ImageDimensions(width=960, height=1344))


def test___len__(dataset_train, dataset_test):
    assert len(dataset_train) == 2
    assert len(dataset_test) == 1


def test__init__test(dataset_test):
    assert len(dataset_test) == 1
    assert dataset_test.image_path_list is not None
    assert dataset_test.output_file_list is not None


def test_get_gt_data_paths(data_dir):
    file_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_dir, True), _get_dataspecs(data_dir, False)])
    assert len(file_list) == 3
    assert file_list[0] == (data_dir / 'codex' / 'D1-LC-Car-folio-1001.jpg',
                            data_dir / 'labels' / 'D1-LC-Car-labA-1001.gif')
    assert file_list[1] == (data_dir / 'codex' / 'D1-LC-Car-folio-1002.jpg',
                            data_dir / 'labels' / 'D1-LC-Car-labA-1002.gif')
    assert file_list[2] == (data_dir / 'codex' / 'D1-LC-Car-folio-1000.jpg',
                            data_dir / 'labels' / 'D1-LC-Car-labA-1000.gif')


def test_get_gt_data_paths_doc_dir_not_found(data_dir, caplog):
    list_specs = [_get_dataspecs(data_dir, True), _get_dataspecs(data_dir, False)]
    list_specs[0].doc_dir = 'nothing'
    with pytest.raises(AssertionError):
        DatasetRolfFormat.get_img_gt_path_list(list_specs=list_specs)
    assert 'Document directory not found' in caplog.text


def test_get_gt_data_paths_gt_dir_not_found(data_dir, caplog):
    list_specs = [_get_dataspecs(data_dir, True), _get_dataspecs(data_dir, False)]
    list_specs[0].gt_dir = 'nothing'
    with pytest.raises(AssertionError):
        DatasetRolfFormat.get_img_gt_path_list(list_specs=list_specs)
    assert 'Ground Truth directory not found' in caplog.text


def test_dataset_rgb_test(dataset_test):
    data_tensor, gt_tensor, idx = dataset_test[0]
    assert data_tensor.shape[-2:] == gt_tensor.shape[-2:]
    assert idx == 0
    assert data_tensor.ndim == 3
    assert gt_tensor.ndim == 3


def test_dataset_rgb_train(dataset_train):
    data_tensor, gt_tensor = dataset_train[0]
    assert data_tensor.shape[-2:] == gt_tensor.shape[-2:]
    assert data_tensor.ndim == 3
    assert gt_tensor.ndim == 3


def test__load_data_and_gt(dataset_train):
    data_img, gt_img = dataset_train._load_data_and_gt(index=0)
    assert data_img.size == gt_img.size
    assert data_img.mode == 'RGB'
    assert gt_img.mode == 'RGB'


def _get_dataspecs(data_root, train: bool = True):
    if train:
        return DatasetSpecs(data_root=data_root,
                            doc_dir="codex",
                            doc_names="D1-LC-Car-folio-####.jpg",
                            gt_dir="labels",
                            gt_names="D1-LC-Car-labA-####.gif",
                            range_from=1001,
                            range_to=1002)
    else:
        return DatasetSpecs(data_root=data_root,
                            doc_dir="codex",
                            doc_names="D1-LC-Car-folio-####.jpg",
                            gt_dir="labels",
                            gt_names="D1-LC-Car-labA-####.gif",
                            range_from=1000,
                            range_to=1000)
