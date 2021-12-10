import math

import src.datamodules.utils.functional


class OneHotToPixelLabelling(object):
    def __call__(self, tensor):
        return src.datamodules.utils.functional.argmax_onehot(tensor)
