from typing import Dict, List

from pytorch_lightning.utilities import LightningEnum


class OutputKeys(LightningEnum):
    PREDICTION = 'pred'
    TARGET = 'target'
    LOG = 'logs'
    LOSS = 'loss'

    def __hash__(self):
        return hash(self.value)


def reduce_dict(input_dict: Dict, key_list: List) -> Dict:
    return {key: value for (key, value) in input_dict.items() if key in key_list}
