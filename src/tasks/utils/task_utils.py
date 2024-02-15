from typing import Callable, Mapping, Sequence, Dict, Union

import pytorch_lightning

from src.utils import utils

log = utils.get_logger(__name__)


# inspired by https://github.com/PyTorchLightning/lightning-flash/blob/2ec52e633bb3679f50dd7e30526885a4547e1851/flash/core/utilities/apply_func.py
def get_callable_name(fn_or_class: Union[Callable, Sequence, object]) -> str:
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


def get_callable_dict(fn: Union[Callable, Mapping, Sequence]) -> Union[Dict, Mapping]:
    if isinstance(fn, Mapping):
        return fn
    elif isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    elif callable(fn):
        return {get_callable_name(fn): fn}


def print_merge_tool_info(trainer: 'pytorch_lightning.Trainer', test_output_path: str, data_format: str):
    datamodule_path = trainer.datamodule.data_dir
    prediction_path = (test_output_path / 'patches').absolute()
    output_path = (test_output_path / 'result').absolute()
    data_folder_name = trainer.datamodule.data_folder_name
    gt_folder_name = trainer.datamodule.gt_folder_name
    log.info(f'To run the merging of patches:')
    log.info(f'python tools/merge_cropped_output_{data_format}.py -d {datamodule_path} -p {prediction_path} '
             f'-o {output_path} -df {data_folder_name} -gf {gt_folder_name}')
