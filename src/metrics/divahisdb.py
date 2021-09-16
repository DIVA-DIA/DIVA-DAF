from typing import Any, Optional, Callable

import torch
import numpy as np
from torchmetrics import Metric


class HisDBIoU(Metric):

    def __init__(self, n_classes: int = None, mask_modifies_prediction: bool = True, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None,
                 ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.n_classes = n_classes
        self.mask_modifies_prediction = mask_modifies_prediction
        # use state save
        self.add_state("tps", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> None:
        # take into account the boundary pixels like done in the offical evaluator
        # https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator/blob/87a11ede232f8fb490401a382b8764697b65ea8d/src/main/java/ch/unifr/LayoutAnalysisEvaluator.java#L225
        if mask is not None:
            if self.mask_modifies_prediction:
                mask = torch.logical_and(mask, torch.eq(pred, 0))
                new_label_trues = label_trues.clone()
                new_label_trues[mask] = label_preds[mask]

        hist = torch.zeros((self.n_classes, self.n_classes)).type_as(new_label_trues)
        for lt, lp in zip(new_label_trues, label_preds):
            try:
                # the images all have the same size
                hist = torch.sum(hist, self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes))
            except ValueError:
                # the images have different sizes
                hist = torch.sum(hist, self._fast_hist([l.flatten() for l in lt].flatten(),
                                                       [l.flatten() for l in lp].flatten(), self.n_classes))

        with np.errstate(divide='ignore', invalid='ignore'):
            self.tps = torch.sum(self.tps, torch.diag(hist))
            self.total = torch.sum(self.total, hist.sum(axis=1) + hist.sum(axis=0) - torch.diag(hist))

    def compute(self) -> Any:
        return self.tps.float() / self.total

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
            torch.sum(torch.mul(n_class, label_true[mask]), label_pred[mask]), minlength=n_class ** 2).reshape(n_class,
                                                                                                               n_class)
        return hist
