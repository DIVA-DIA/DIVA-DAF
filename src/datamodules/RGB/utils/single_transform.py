import src.datamodules.RGB.utils.functional as F_custom


class IntegerEncoding(object):
    def __init__(self, class_encodings):
        """
        Convert ground truth tensor to integer encoded matrix.

        :param class_encodings: class encoding so which class (index) has what value (element)
        :type class_encodings: List[int]
        """
        self.class_encodings = class_encodings

    def __call__(self, gt):
        """
        :param gt: ground truth Tensor
        :type gt: torch.Tensor
        :return: Integer encoded ground truth
        :rtype: torch.Tensor
        """


        return F_custom.gt_to_int_encoding(gt, self.class_encodings)


class OneHotEncoding(object):
    def __init__(self, class_encodings):
        """
        Convert ground truth tensor to one-hot encoded matrix.

        :param class_encodings: class encoding so which class (index) has what value (element)
        :type class_encodings: List[int]
        """
        self.class_encodings = class_encodings

    def __call__(self, gt):
        """
        :param gt: ground truth Tensor
        :type gt: torch.Tensor
        :return: One-hot encoded ground truth
        :rtype: torch.Tensor
        """
        return F_custom.gt_to_one_hot(gt, self.class_encodings)
