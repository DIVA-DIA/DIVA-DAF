import logging

import pytest
import time

from src.callbacks.general_callbacks import TimeTracker


@pytest.fixture
def time_tracker():
    return TimeTracker()


def test_init_callback(time_tracker):
    assert time_tracker.start_time_train is None
    assert time_tracker.start_time_train_epoch is None
    assert time_tracker.start_time_test is None


def test_on_train_start(time_tracker):
    time_tracker.on_train_start(None, None)
    assert time_tracker.start_time_train is not None
    assert time_tracker.start_time_train > 0
    assert time_tracker.start_time_train_epoch is None
    assert time_tracker.start_time_test is None


def test_on_train_epoch_start(time_tracker):
    time_tracker.on_train_epoch_start(None, None)
    assert time_tracker.start_time_train_epoch is not None
    assert time_tracker.start_time_train_epoch > 0
    assert time_tracker.start_time_train is None
    assert time_tracker.start_time_test is None


def test_on_test_epoch_start(time_tracker):
    time_tracker.on_test_epoch_start(None, None)
    assert time_tracker.start_time_test is not None
    assert time_tracker.start_time_test > 0
    assert time_tracker.start_time_train is None
    assert time_tracker.start_time_train_epoch is None


def test_on_train_epoch_end(time_tracker, caplog, mocker):
    time_tracker.start_time_train_epoch = 10
    mocker.patch('time.time', return_value=15)
    log = logging.getLogger("__name__")
    with caplog.at_level(logging.INFO):
        time_tracker.on_train_epoch_end = lambda *args: log.info("time: " + str(time.time()-time_tracker.start_time_train_epoch))
        time_tracker.on_train_epoch_end(None, None)
    assert "time: 5" in caplog.text


def test_on_test_epoch_end(time_tracker, caplog, mocker):
    time_tracker.start_time_test = 20
    mocker.patch('time.time', return_value=35)
    log = logging.getLogger("__name__")
    with caplog.at_level(logging.INFO):
        time_tracker.on_test_epoch_end = lambda *args: log.info("time: " + str(time.time()-time_tracker.start_time_test))
        time_tracker.on_test_epoch_end(None, None)
    assert "time: 15" in caplog.text
