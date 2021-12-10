import random

import numpy as np
import pytest
import torch
from torchvision.datasets.folder import pil_loader

from src.datamodules.utils.twin_transforms import TwinRandomCrop, TwinImageToTensor, TwinCompose, \
    ToTensorSlidingWindowCrop
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


def test_twin_compose(img_and_gt):
    img, gt = img_and_gt
    random.seed(42)
    crop_size = 2
    transformation1 = TwinRandomCrop(crop_size=crop_size)
    transformation2 = TwinImageToTensor()
    compose = TwinCompose([transformation1, transformation2])
    trans_img, trans_gt = compose(img=img, gt=gt)
    assert isinstance(trans_img, torch.Tensor)
    assert isinstance(trans_gt, torch.Tensor)
    assert trans_img.shape == torch.Size([3, crop_size, crop_size])
    assert trans_gt.shape == torch.Size([3, crop_size, crop_size])


def test_twin_random_crop(img_and_gt):
    img, gt = img_and_gt
    random.seed(42)
    crop_size = 2
    transformation = TwinRandomCrop(crop_size=crop_size)
    trans_img, trans_gt = transformation(img=img, gt=gt)
    assert trans_img.width == crop_size and trans_img.height == crop_size
    assert trans_gt.width == crop_size and trans_gt.height == crop_size
    assert np.array_equal(np.array(trans_img), [[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]]])
    assert np.array_equal(np.array(trans_gt), [[[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]])


def test_twin_image_to_tensor(img_and_gt):
    img, gt = img_and_gt
    transformation = TwinImageToTensor()
    trans_img, trans_gt = transformation(img=img, gt=gt)
    assert isinstance(trans_img, torch.Tensor)
    assert isinstance(trans_gt, torch.Tensor)


def test_to_tensor_sliding_window_crop(img_and_gt):
    img, gt = img_and_gt
    crop_size = 2
    coordinates = [0, 0]
    transform = ToTensorSlidingWindowCrop(crop_size=crop_size)
    trans_img, trans_gt = transform(img=img, gt=gt, coordinates=coordinates)
    assert isinstance(trans_img, torch.Tensor)
    assert isinstance(trans_gt, torch.Tensor)
    assert torch.equal(trans_img, torch.tensor([[[0.0078431377, 0.0078431377], [0.0078431377, 0.0078431377]],
                                                [[0.0078431377, 0.0078431377], [0.0078431377, 0.0078431377]],
                                                [[0.0078431377, 0.0078431377], [0.0078431377, 0.0078431377]]]))
    assert torch.equal(trans_gt, torch.tensor([[[0.0000000000, 0.0000000000], [0.0000000000, 0.0000000000]],
                                               [[0.0000000000, 0.0000000000], [0.0000000000, 0.0000000000]],
                                               [[0.0039215689, 0.0039215689], [0.0039215689, 0.0039215689]]]))


@pytest.fixture()
def img_and_gt(data_dir_cropped):
    folder_name = 'e-codices_fmb-cb-0055_0098v_max'
    file_name = 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000.png'
    img_path = data_dir_cropped / 'train' / 'data' / folder_name / file_name
    gt_path = data_dir_cropped / 'train' / 'gt' / folder_name / file_name
    return pil_loader(img_path), pil_loader(gt_path)
