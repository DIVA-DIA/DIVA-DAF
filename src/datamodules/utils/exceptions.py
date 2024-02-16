class PathNone(Exception):
    """
    Raised when a path is None
    """
    pass


class PathNotDir(Exception):
    """
    Raised when a path is not a directory
    """
    pass


class PathMissingSplitDir(Exception):
    """
    Raised when a path is missing a split directory
    """
    pass


class PathMissingDirinSplitDir(Exception):
    """
    Raised when a path is missing a directory in a split directory
    """
    pass
