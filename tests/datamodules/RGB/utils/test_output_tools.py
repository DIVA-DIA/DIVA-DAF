import numpy as np
import pytest
import torch
from PIL import Image

from src.datamodules.RGB.utils.output_tools import output_to_class_encodings, save_output_page_image


@pytest.fixture()
def input_image():
    return torch.tensor([[[0., 0.3], [4., 2.]],
                         [[1., 4.1], [-0.2, 1.9]],
                         [[1.1, -0.8], [4.9, 1.3]],
                         [[-0.4, 4.4], [2.9, 0.1]]])


@pytest.fixture()
def class_encodings():
    return [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]


def test_save_output_page_image(tmp_path, input_image, class_encodings):
    img_name = 'test.png'
    save_output_page_image(img_name, input_image, tmp_path, class_encodings)
    img_output_path = tmp_path / img_name
    loaded_img = Image.open(img_output_path)
    assert img_output_path.exists()
    assert np.array_equal(output_to_class_encodings(input_image, class_encodings), np.array(loaded_img))


def test_output_to_class_encodings(input_image, class_encodings):
    encoded = output_to_class_encodings(output=input_image, class_encodings=class_encodings)
    expected_output = [[[255, 255, 0], [255, 255, 255]], [[255, 255, 0], [0, 0, 0]]]
    assert np.array_equal(encoded, expected_output)
