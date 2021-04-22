import numpy as np
from PIL import Image
from torch import tensor, equal

from src.models.semantic_segmentation.utils.output_tools import _get_argmax, merge_patches, output_to_class_encodings, \
    save_output_page_image

# batchsize (2) x classes (4) x W (2) x H (2)
BATCH = tensor([[[[0., 0.3], [4., 2.]],
                 [[1., 4.1], [-0.2, 1.9]],
                 [[1.1, -0.8], [4.9, 1.3]],
                 [[-0.4, 4.4], [2.9, 0.1]]],
                [[[0.1, 3.1], [5.3, 0.8]],
                 [[2.4, 7.4], [-2.2, 0.2]],
                 [[2.6, 0.7], [-4.3, 1.2]],
                 [[1.0, -2.4], [-0.8, 1.8]]]])
CLASS_ENCODINGS = [1., 2., 4., 8.]


def test__get_argmax():
    argmax = _get_argmax(BATCH)
    result = tensor([[[2, 3],
                      [2, 0]],
                     [[2, 1],
                      [0, 3]]])
    assert equal(result, argmax)


def test_merge_patches_one_image():
    patch = BATCH[0]
    output_img = np.empty((4, 2, 2))
    output_img[:] = np.nan
    merged_output = merge_patches(patch, (0, 0), output_img)
    patch = patch.cpu().numpy()
    assert merged_output.shape == patch.shape
    assert np.array_equal(merged_output, patch)


def test_merge_patches_two_images():
    output_img = np.empty((4, 2, 3))
    output_img[:] = np.nan
    coordiantes = [(0, 0), (0, 1)]
    for patch, coords in zip(BATCH, coordiantes):
        merge_patches(patch, coords, output_img)
    expected_output = np.array(
        [[[0.0, 0.30000001192092896, 3.0999999046325684], [4.0, 5.300000190734863, 0.800000011920929]],
         [[1.0, 4.099999904632568, 7.400000095367432], [-0.20000000298023224, 1.899999976158142, 0.20000000298023224]],
         [[1.100000023841858, 2.5999999046325684, 0.699999988079071],
          [4.900000095367432, 1.2999999523162842, 1.2000000476837158]],
         [[-0.4000000059604645, 4.400000095367432, -2.4000000953674316],
          [2.9000000953674316, 0.10000000149011612, 1.7999999523162842]]])
    assert expected_output.shape == output_img.shape
    assert np.array_equal(expected_output, output_img)


def test_save_output(tmp_path):
    img = BATCH[0]
    img_name = 'test.png'
    save_output_page_image(img_name, img, tmp_path, CLASS_ENCODINGS)
    img_output_path = tmp_path / 'images' / ('output_' + img_name)
    loaded_img = Image.open(img_output_path)
    assert img_output_path.exists()
    assert np.array_equal(output_to_class_encodings(img, CLASS_ENCODINGS), np.array(loaded_img))


def test_output_to_class_encodings():
    img = BATCH[0]
    encoded_img = output_to_class_encodings(img, CLASS_ENCODINGS)
    expected_output = np.array([[[0, 0, 4], [0, 0, 8]], [[0, 0, 4], [0, 0, 1]]])
    assert encoded_img.shape == expected_output.shape
    assert np.array_equal(expected_output, encoded_img)
