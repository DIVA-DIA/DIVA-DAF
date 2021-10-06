import numpy as np
import torch

from src.metrics.divahisdb import HisDBIoU


def test_iou_boundary_mask_modifies_prediction_identical():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=True, identical=True)
    metric = HisDBIoU(num_classes=num_classes)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.tensor(1.) == iou


def test_iou_mask_modifies_prediction_identical():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=False, identical=True)
    metric = HisDBIoU(num_classes=num_classes)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.tensor(1.) == iou


def test_iou_boundary_mask_modifies_prediction():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=True, identical=False)
    metric = HisDBIoU(num_classes=num_classes)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.isclose(iou, torch.tensor((12 / 13 + 5 / 6) / 2))


def test_iou_boundary_mask_modifies():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=False, identical=False)
    metric = HisDBIoU(num_classes=num_classes)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.isclose(iou, torch.tensor((12 / 14 + 4 / 6) / 2))


# with mask modifies prediction false

def test_iou_boundary_mask_modifies_prediction_false_identical():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=True, identical=True)
    metric = HisDBIoU(num_classes=num_classes, mask_modifies_prediction=False)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.tensor(1.) == iou


def test_iou_mask_modifies_prediction_false_identical():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=False, identical=True)
    metric = HisDBIoU(num_classes=num_classes, mask_modifies_prediction=False)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.tensor(1.) == iou


def test_iou_boundary_mask_modifies_false_prediction():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=True, identical=False)
    metric = HisDBIoU(num_classes=num_classes, mask_modifies_prediction=False)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.isclose(iou, torch.tensor((13 / 14 + 4 / 5) / 2))


def test_iou_boundary_mask_modifies():
    label_preds, label_trues, num_classes, mask = _get_test_data(with_boundary=False, identical=False)
    metric = HisDBIoU(num_classes=num_classes, mask_modifies_prediction=False)
    metric.update(pred=label_preds, target=label_trues, mask=mask)
    iou = metric.compute()

    assert torch.isclose(iou, torch.tensor((12 / 14 + 4 / 6) / 2))


def test__fast_hist():
    label_preds, label_trues, num_classes, _ = _get_test_data()
    output = HisDBIoU._fast_hist(label_trues, label_preds, num_classes)
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
    num_classes = len(label_trues.unique())
    mask = torch.tensor([[[False, False], [False, True], [False, False]],
                         [[False, False], [False, False], [False, False]],
                         [[False, False], [False, False], [False, False]]], device=device)
    if not with_boundary:
        mask[:] = False
    return label_preds, label_trues, num_classes, mask
