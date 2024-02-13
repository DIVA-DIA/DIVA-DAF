from typing import Dict, List

from pytorch_lightning.utilities import LightningEnum


class OutputKeys(LightningEnum):
    """
    Enum class for the keys of the output dictionary.
    """
    PREDICTION = 'pred'
    TARGET = 'target'
    LOG = 'logs'
    LOSS = 'loss'

    def __hash__(self):
        return hash(self.value)


def reduce_dict(input_dict: Dict, key_list: List) -> Dict:
    """
    Reduce the input dictionary to only contain the keys in the key_list.

    :param input_dict: The dictionary to reduce
    :type input_dict: Dict
    :param key_list: List of keys to keep
    :type key_list: List
    :return: The dictionary input_dict with only the keys in the key_list
    :rtype: Dict
    """
    return {key: input_dict[key] for key in key_list if key in input_dict}
