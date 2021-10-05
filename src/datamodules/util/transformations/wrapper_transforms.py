from typing import Callable


class OnlyImage(object):
    """Wrapper function around a single parameter transform. It will be cast only on image"""

    def __init__(self, transform: Callable):
        """Initialize the transformation with the transformation to be called.
        Could be a compose.

        Parameters
        ----------
        transform : torchvision.transforms.transforms
            Transformation to wrap
        """
        self.transform = transform

    def __call__(self, image, target):
        return self.transform(image), target


class OnlyTarget(object):
    """Wrapper function around a single parameter transform. It will be cast only on target"""

    def __init__(self, transform: Callable):
        """Initialize the transformation with the transformation to be called.
        Could be a compose.

        Parameters
        ----------
        transform : torchvision.transforms.transforms
            Transformation to wrap
        """
        self.transform = transform

    def __call__(self, image, target):
        return image, self.transform(target)