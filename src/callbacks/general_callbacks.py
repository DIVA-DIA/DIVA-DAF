from time import time

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import Callback


class TimeTracker(Callback):
    """
    A callback to track the time taken for training and testing.
    It logs the time taken for each epoch and the total time taken for training and testing.
    """
    def __init__(self):
        self.start_time_train = None
        self.start_time_train_epoch = None
        self.start_time_test = None

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.start_time_train = time()

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time_train_epoch = time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1:
            self.log("train/total_time", time() - self.start_time_train)
        self.log("train/epoch_time", time() - self.start_time_train_epoch)

    @rank_zero_only
    def on_test_epoch_start(self, trainer, pl_module):
        self.start_time_test = time()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.log("test/total_time", time() - self.start_time_test)
