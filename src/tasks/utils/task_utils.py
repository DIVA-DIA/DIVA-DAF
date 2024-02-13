from typing import Callable, Mapping, Sequence, Dict, Union


# inspired by https://github.com/PyTorchLightning/lightning-flash/blob/2ec52e633bb3679f50dd7e30526885a4547e1851/flash/core/utilities/apply_func.py
def get_callable_name(fn_or_class: Union[Callable, Sequence, object]) -> str:
    """
    Get the name of a callable or class.

    :param fn_or_class: Callable, class or sequence we want the name of
    :type fn_or_class: Union[Callable, Sequence, object]
    :return: the name of the callable or class
    :rtype: str
    """
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


def get_callable_dict(fn: Union[Callable, Mapping, Sequence]) -> Union[Dict, Mapping]:
    """
    Creates a dictionary with the name of the callable as key and the callable as value.

    :param fn: Callable, sequence or mapping we want to convert to a dictionary
    :type fn: Union[Callable, Mapping, Sequence]
    :return: A dictionary with the name of the callable as key and the callable as value
    :rtype: Union[Dict, Mapping]
    """
    if isinstance(fn, Mapping):
        return fn
    elif isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    elif callable(fn):
        return {get_callable_name(fn): fn}
