import numpy as np
import torch

from src.utils.accuracy import accuracy_segmentation, _fast_hist


def test_accuracy_segmentation_boundary_identical():
    label_preds, label_trues, nb_classes, mask = _get_test_data(with_boundary=True, identical=True)
    accuracy, mean_accuracy, miou, fwavacc = accuracy_segmentation(label_trues, label_preds, nb_classes, mask)
    assert accuracy == 100.
    assert mean_accuracy == 100.0
    assert miou == 100.
    assert fwavacc == 100.


def test_accuracy_segmentation_identical():
    label_preds, label_trues, nb_classes, mask = _get_test_data(with_boundary=False, identical=True)
    accuracy, mean_accuracy, miou, fwavacc = accuracy_segmentation(label_trues, label_preds, nb_classes, mask)
    assert accuracy == 100.
    assert mean_accuracy == 100.0
    assert miou == 100.
    assert fwavacc == 100.


def test_accuracy_segmentation_boundary():
    label_preds, label_trues, nb_classes, mask = _get_test_data(with_boundary=True, identical=False)
    accuracy, mean_accuracy, miou, fwavacc = accuracy_segmentation(label_trues, label_preds, nb_classes, mask)
    assert np.isclose(accuracy.numpy(), 100 * 17 / 18)
    assert np.isclose(mean_accuracy, 100 * (13 / 13 + 4 / 5) / 2)
    assert np.isclose(miou, 100 * (13 / 14 + 4 / 5) / 2)
    assert np.isclose(fwavacc, 100 * (((13 / 18) * (13 / 14)) + ((5 / 18) * (4 / 5))))


def test_accuracy_segmentation():
    label_preds, label_trues, nb_classes, mask = _get_test_data(with_boundary=False, identical=False)
    accuracy, mean_accuracy, miou, fwavacc = accuracy_segmentation(label_trues, label_preds, nb_classes, mask)
    assert np.isclose(accuracy.numpy(), 100 * 16 / 18)
    assert np.isclose(mean_accuracy, 100 * (12 / 12 + 4 / 6) / 2)
    assert np.isclose(miou, 100 * (12 / 14 + 4 / 6) / 2)
    assert np.isclose(fwavacc, 100 * (((12 / 18) * (12 / 14)) + ((6 / 18) * (4 / 6))))


def test_accuracy_segmentation_identical_mask_None():
    label_preds, label_trues, nb_classes, _ = _get_test_data(identical=True)
    accuracy, mean_accuracy, miou, fwavacc = accuracy_segmentation(label_trues, label_preds, nb_classes, None)
    assert accuracy == 100.
    assert mean_accuracy == 100.0
    assert miou == 100.
    assert fwavacc == 100.


def test_accuracy_segmentation():
    label_preds, label_trues, nb_classes, _ = _get_test_data(with_boundary=False, identical=False)
    accuracy, mean_accuracy, miou, fwavacc = accuracy_segmentation(label_trues, label_preds, nb_classes, None)
    assert np.isclose(accuracy.numpy(), 100 * 16 / 18)
    assert np.isclose(mean_accuracy, 100 * (12 / 12 + 4 / 6) / 2)
    assert np.isclose(miou, 100 * (12 / 14 + 4 / 6) / 2)
    assert np.isclose(fwavacc, 100 * (((12 / 18) * (12 / 14)) + ((6 / 18) * (4 / 6))))


def test__fast_hist():
    label_preds, label_trues, nb_classes, _ = _get_test_data()
    output = _fast_hist(label_trues, label_preds, nb_classes)
    expected_result = torch.tensor([[12, 0], [0, 6]])
    assert torch.equal(expected_result, output)


def _get_test_data(with_boundary=True, identical=True):
    """
    Produces test data in the format [Batch size x W x H], where batch size is 2, W=3 and H=3.
    In the first batch the pixel at 3,2 is a boundary pixel and in the second pixel at 1,2

    :param with_boundary: bool
        if true there is a boundary pixel like described
    :param identical: bool
        if true pred and trues are the same
    :return:
    """
    device = torch.device('cpu')
    # 0 = Background; 1 = Text
    # 0, 0, 0   ; 0, (1), 0
    # 0, 1, 0   ; 0, 0, 0
    # 0, 1, 0 ; 1, 1, 1
    # batch size 2; 3x3; () --> masked
    label_trues = torch.tensor([[[0, 0], [0, 1], [0, 0]],
                                [[0, 0], [1, 0], [0, 0]],
                                [[0, 1], [1, 1], [0, 1]]], device=device)
    # 0, 0, 0   ; 0, [0], 0
    # 0, 1, 0   ; 0, 0, 0
    # 0, [0], 0   ; 1, 1, 1
    label_preds = torch.tensor([[[0, 0], [0, 1], [0, 0]],
                                [[0, 0], [1, 0], [0, 0]],
                                [[0, 1], [1, 1], [0, 1]]], device=device)
    if not identical:
        label_preds[0, 1, 1] = 0
        label_preds[2, 1, 0] = 0
    nb_classes = len(label_trues.unique())
    mask = torch.tensor([[[False, False], [False, True], [False, False]],
                         [[False, False], [False, False], [False, False]],
                         [[False, False], [False, False], [False, False]]], device=device)
    if not with_boundary:
        mask[:] = False
    return label_preds, label_trues, nb_classes, mask
