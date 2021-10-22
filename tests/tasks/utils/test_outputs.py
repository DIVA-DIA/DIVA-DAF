from _pytest.fixtures import fixture

from src.tasks.utils.outputs import OutputKeys, reduce_dict


@fixture
def get_dict():
    return {OutputKeys.PREDICTION: [1, 2, 3, 3],
            OutputKeys.TARGET: [1, 2, 3, 4],
            OutputKeys.LOSS: 0.537284,
            OutputKeys.LOG: {'iou': 0.98245, 'precision': 0.897824, 'crossentropyloss': 0.537284}}


def test_reduce_dict_full(get_dict):
    result = reduce_dict(input_dict=get_dict, key_list=[OutputKeys.PREDICTION,
                                                        OutputKeys.TARGET,
                                                        OutputKeys.LOSS,
                                                        OutputKeys.LOG])
    assert OutputKeys.PREDICTION in result
    assert OutputKeys.TARGET in result
    assert OutputKeys.LOSS in result
    assert OutputKeys.LOG in result
    assert result[OutputKeys.PREDICTION] == [1, 2, 3, 3]
    assert result[OutputKeys.TARGET] == [1, 2, 3, 4]
    assert result[OutputKeys.LOSS] == 0.537284
    assert result[OutputKeys.LOG] == {'iou': 0.98245, 'precision': 0.897824, 'crossentropyloss': 0.537284}


def test_reduce_dict(get_dict):
    result = reduce_dict(input_dict=get_dict, key_list=[OutputKeys.PREDICTION,
                                                        OutputKeys.TARGET,
                                                        ])
    assert OutputKeys.PREDICTION in result
    assert OutputKeys.TARGET in result
    assert result[OutputKeys.PREDICTION] == [1, 2, 3, 3]
    assert result[OutputKeys.TARGET] == [1, 2, 3, 4]


def test_reduce_dict_empty_list(get_dict):
    result = reduce_dict(input_dict=get_dict, key_list=[])
    assert not bool(result)
    assert OutputKeys.PREDICTION not in result
    assert OutputKeys.TARGET not in result
    assert OutputKeys.LOSS not in result
    assert OutputKeys.LOG not in result
