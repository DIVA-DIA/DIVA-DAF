from enum import Enum, auto


class GT_Type(Enum):
    CLASSIFICATION = auto()
    VECTOR = auto()
    ROW_COLUMN = auto()
    FULL_IMAGE = auto()
