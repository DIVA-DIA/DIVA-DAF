import numpy as np
import torch


def accuracy_segmentation(label_trues, label_preds, n_class, mask,
                          calc_acc=False, calc_acc_cls=False, calc_mean_iu=False, calc_fwavacc=False):
    """
    Taken from https://github.com/wkentaro/pytorch-fcn
    Calculates the accuracy measures for the segmentation runner.
    It uses the boundary pixel system introduced in the DIVA hisdb dataset.


    :param label_trues: matrix (batch size x H x W)
        contains the true class labels for each pixel
    :param label_preds: matrix (batch size x H x W)
        contains the predicted class for each pixel
    :param n_class: int
        number possible classes
    :param mask: matrix [bool] (batch size x H x W)
        the boundary pixel mask. If None it does not take the mask into account
    :return: (tensor, tensor, float, float)
        (overall accuracy, mean accuracy, mean IU, fwavacc)
    """

    # Initialize all metrics with -1
    acc = acc_cls = mean_iu = fwavacc = -1

    # If nothing to do, return -100 for all metrics
    if not calc_acc and not calc_acc_cls and not calc_mean_iu and not calc_fwavacc:
        return acc * 100, acc_cls * 100, mean_iu * 100, fwavacc * 100

    # take into account the boundary pixels like done in the offical evaluator
    # https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator/blob/87a11ede232f8fb490401a382b8764697b65ea8d/src/main/java/ch/unifr/LayoutAnalysisEvaluator.java#L225
    new_label_trues = label_trues
    if mask is not None:
        mask = torch.logical_and(mask, label_preds == 0)
        new_label_trues = label_trues.clone()
        new_label_trues[mask] = label_preds[mask]

    hist = torch.zeros((n_class, n_class)).type_as(new_label_trues)
    for lt, lp in zip(new_label_trues, label_preds):
        try:
            # the images all have the same size
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        except ValueError:
            # the images have different sizes
            hist += _fast_hist([l.flatten() for l in lt].flatten(), [l.flatten() for l in lp].flatten(), n_class)

    if calc_acc:
        acc = torch.diag(hist).sum() / hist.sum()

    if calc_acc_cls:
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = torch.diag(hist) / hist.sum(axis=1)
        acc_cls = acc_cls[~torch.isnan(acc_cls)].mean()

    if calc_mean_iu or calc_fwavacc:
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = torch.diag(hist) / (
                    hist.sum(axis=1) + hist.sum(axis=0) - torch.diag(hist)
            )

    if calc_mean_iu :
        mean_iu = iu[~torch.isnan(iu)].mean()

    if calc_fwavacc:
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc * 100, acc_cls * 100, mean_iu * 100, fwavacc * 100


def _fast_hist(label_true, label_pred, n_class):
    """
    Taken from https://github.com/wkentaro/pytorch-fcn

    :param label_true: matrix (batch size x H x W)
        contains the true class labels for each pixel
    :param label_pred: matrix (batch size x H x W)
        contains the predicted class for each pixel
    :param n_class: int
        number possible classes

    :return
        histogram
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask] +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
