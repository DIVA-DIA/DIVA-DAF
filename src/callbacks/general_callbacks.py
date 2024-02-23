from datetime import time

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import Callback


class TimeTracker(Callback):
    def __init__(self):
        self.start_time_train = None
        self.start_time_train_epoch = None
        self.start_time_test = None

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.start_time_train = time()

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        self.log("train/total_time", time() - self.start_time_train)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time_train_epoch = time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.log("train/epoch_time", time() - self.start_time_train_epoch)

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        self.start_time_test = time()

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        self.log("test/total_time", time() - self.start_time_test)
