from typing import Dict, List

import numpy
import numpy as np
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


def save_numpy_files(trainer, test_output_path, input_idx, output):
    if not hasattr(trainer.datamodule, 'get_img_name_coordinates'):
        raise NotImplementedError('Datamodule does not provide detailed information of the crop')
    for patch, idx in zip(output[OutputKeys.PREDICTION].detach().cpu().numpy(),
                          input_idx.detach().cpu().numpy()):
        patch_info = trainer.datamodule.get_img_name_coordinates(idx)
        img_name = patch_info[0]
        patch_name = patch_info[1]
        dest_folder = test_output_path / 'patches' / img_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_filename = dest_folder / f'{patch_name}.npy'

        np.save(file=str(dest_filename), arr=patch)
