from typing import Callable


class OnlyImage(object):
    """
    Wrapper function around a single parameter transform. It will be cast only on image

    :param transform: Transformation to apply to the codex image
    :type transform: Callable

    """

    def __init__(self, transform: Callable):
        """
        Constructor method for the OnlyImage class.
        """
        self.transform = transform

    def __call__(self, image, target):
        return self.transform(image), target


class OnlyTarget(object):
    """
    Wrapper function around a single parameter transform. It will be cast only on target

    :param transform: Transformation to apply to the ground truth image
    :type transform: Callable
    """

    def __init__(self, transform: Callable):
        """
        Constructor method for the OnlyTarget class.
        """
        self.transform = transform

    def __call__(self, image, target):
        return image, self.transform(target)