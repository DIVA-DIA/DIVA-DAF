from typing import Any, Optional, Callable

import numpy as np
import torch
from torchmetrics import Metric


class HisDBIoU(Metric):
    """
    Implementation of the mIoU metric used in the paper.Dataset implementation of the RotNet paper of `Gidaris et al. <https://arxiv.org/abs/1803.07728>`_.

    """

    def __init__(self, num_classes: int = None, mask_modifies_prediction: bool = True, compute_on_step: bool = True,
                 dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None,
                 ) -> None:
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.num_classes = num_classes
        self.mask_modifies_prediction = mask_modifies_prediction
        # use state save
        self.add_state("tps", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> None:
        # take into account the boundary pixels like done in the offical evaluator
        # https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator/blob/87a11ede232f8fb490401a382b8764697b65ea8d/src/main/java/ch/unifr/LayoutAnalysisEvaluator.java#L225
        if mask is not None:
            mask_and_bg_predicted = torch.logical_and(mask, torch.eq(pred, 0))
            if self.mask_modifies_prediction:
                pred = pred.clone()
                pred[mask_and_bg_predicted] = target[mask_and_bg_predicted]
            else:
                target = target.clone()
                target[mask_and_bg_predicted] = pred[mask_and_bg_predicted]

        hist = torch.zeros((self.num_classes, self.num_classes)).type_as(target)
        for lt, lp in zip(target, pred):
            try:
                # the images all have the same size
                hist = torch.add(hist, self._fast_hist(lt.flatten(), lp.flatten(), self.num_classes))
            except ValueError:
                # the images have different sizes
                hist = torch.add(hist, self._fast_hist([l.flatten() for l in lt].flatten(),
                                                       [l.flatten() for l in lp].flatten(), self.num_classes))

        with np.errstate(divide='ignore', invalid='ignore'):
            self.tps = torch.add(self.tps, torch.diag(hist))
            self.total = torch.add(self.total, hist.sum(axis=1) + hist.sum(axis=0) - torch.diag(hist))

    def compute(self) -> Any:
        res = torch.div(self.tps.float(), self.total)
        return res[~res.isnan()].mean()

    @staticmethod
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
        mask = torch.bitwise_and(torch.ge(label_true, 0), torch.lt(label_true, n_class))
        hist = torch.bincount(
            torch.add(torch.mul(n_class, label_true[mask]), label_pred[mask]), minlength=n_class ** 2).reshape(n_class,
                                                                                                               n_class)
        return hist
