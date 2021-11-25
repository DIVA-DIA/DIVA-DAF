import pytest
import torch
from torchvision.transforms import ToTensor

from src.datamodules.utils.dataset_predict import DatasetPredict
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir


@pytest.fixture
def file_path_list(data_dir):
    test_data_path = data_dir / 'test' / 'data'
    return [str(p) for p in test_data_path.iterdir()]


@pytest.fixture
def predict_dataset(file_path_list):
    return DatasetPredict(image_path_list=file_path_list, image_dims=ImageDimensions(width=487, height=649))


def test__load_data_and_gt(predict_dataset):
    img = predict_dataset._load_data_and_gt(index=0)
    assert img.size == (487, 649)
    assert img.mode == 'RGB'
    assert torch.equal(ToTensor()(img),  predict_dataset[0][0])


def test__apply_transformation(predict_dataset):
    img = predict_dataset._load_data_and_gt(index=0)
    img_tensor = predict_dataset._apply_transformation(img)
    assert torch.equal(img_tensor, predict_dataset[0][0])
    assert img_tensor.shape == torch.Size((3, 649, 487))
