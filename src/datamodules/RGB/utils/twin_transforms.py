from src.datamodules.RGB.utils import functional as F_custom


class IntegerEncoding(object):
    def __init__(self, class_encodings):
        self.class_encodings = class_encodings

    def __call__(self, gt):
        """
        Args:

        Returns:

        """
        return F_custom.gt_to_int_encoding(gt, self.class_encodings)
