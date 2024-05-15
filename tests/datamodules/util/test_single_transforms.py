import torch

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from src.datamodules.utils.single_transforms import OneHotToPixelLabelling, RightAngleRotation, MorphoBuilding
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir


def test_one_hot_to_pixel_labelling():
    transformation = OneHotToPixelLabelling()
    tensor_input = torch.tensor([[[0.6999015212, 0.4833144546],
                                  [0.8329959512, 0.1569360495]],
                                 [[0.5571944118, 0.1454262733],
                                  [0.4641100168, 0.0191639662]],
                                 [[0.8016914725, 0.2552427053],
                                  [0.6105983257, 0.2813706994]]])
    trans_output = transformation(tensor_input)
    assert isinstance(trans_output, torch.Tensor)
    assert torch.equal(trans_output, torch.tensor([[2, 0], [0, 2]]))


def test_right_angle_rotation(mocker):
    mocker.patch('torch.randint', return_value=torch.tensor([1]))
    transformation = RightAngleRotation()
    tensor_input = torch.Tensor([[[0.5, 0.5],
                                  [0.1, 0.1]],
                                 [[0.2, 0.2],
                                  [0.9, 0.9]]])
    trans_output = transformation(tensor=tensor_input)
    assert transformation.target_class == 1
    assert torch.equal(trans_output, torch.tensor([[[0.5, 0.1],
                                                    [0.5, 0.1]],
                                                   [[0.2, 0.9],
                                                    [0.2, 0.9]]]))


def test__update_target_class():
    transformation = RightAngleRotation()
    assert transformation.target_class is None
    transformation._update_target_class()
    assert transformation.target_class is not None


def test_morpho():
    assert False


def test__dilation(data_dir):
    trans = MorphoBuilding(first_filter_size=(1, 45), second_filter_size=(25, 25), border_size=10)
    img = Image.open(next((data_dir / 'train' / 'data').iterdir()))
    morph_img_tensor = trans._morpho(img)

    ToPILImage()(morph_img_tensor[0][0, :, :, :]).save('test_1.png')
    ToPILImage()(morph_img_tensor[1][0, :, :, :]).save('test_2.png')

    assert False
