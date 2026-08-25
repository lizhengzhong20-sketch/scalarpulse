from __future__ import annotations

import math
import sys

import pytest

from scalarpulse._util import to_scalar
from scalarpulse.torch import TorchLogger, gradient_norm, learning_rates


class FakeTensor:
    def __init__(self, value, calls=None):
        self.value = value
        self.calls = calls if calls is not None else []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def item(self):
        self.calls.append("item")
        return self.value


class RecordingTracker:
    def __init__(self):
        self.calls = []

    def log(self, metrics, *, step=None):
        self.calls.append((metrics, step))


class FakeGradient:
    def __init__(self, norm_value):
        self.norm_value = norm_value
        self.norm_types = []
        self.detached = False

    def detach(self):
        self.detached = True
        return self

    def norm(self, norm_type):
        self.norm_types.append(norm_type)
        return FakeTensor(self.norm_value)


class FakeParameter:
    def __init__(self, gradient):
        self.grad = gradient


class FakeModel:
    def __init__(self, gradients):
        self.gradients = gradients

    def parameters(self):
        return [FakeParameter(gradient) for gradient in self.gradients]


class FakeOptimizer:
    param_groups = [{"lr": 0.1}, {"lr": "0.01"}, {"momentum": 0.9}]


def test_import_and_scalar_conversion_do_not_require_pytorch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    calls = []
    tensor = FakeTensor(1.25, calls)

    assert to_scalar(tensor, name="loss") == 1.25
    assert calls == ["detach", "cpu", "item"]


def test_torch_logger_builds_prefixed_payload_with_loss_metrics_and_learning_rates():
    tracker = RecordingTracker()
    logger = TorchLogger(tracker, optimizer=FakeOptimizer(), prefix="/train/")
    loss = FakeTensor(2.5)

    logger.log(loss=loss, metrics={"accuracy": 0.75}, step=8)

    assert tracker.calls == [
        (
            {
                "train": {
                    "accuracy": 0.75,
                    "loss": loss,
                    "lr": 0.1,
                    "lr/group_1": 0.01,
                }
            },
            8,
        )
    ]


def test_learning_rates_requires_param_groups_and_uses_stable_names():
    assert learning_rates(FakeOptimizer()) == {"lr": 0.1, "lr/group_1": 0.01}
    with pytest.raises(TypeError, match="param_groups"):
        learning_rates(object())


def test_gradient_norm_aggregates_detached_parameter_norms_without_torch():
    first = FakeGradient(3.0)
    second = FakeGradient(4.0)
    model = FakeModel([first, None, second])

    assert gradient_norm(model) == 5.0
    assert first.detached and second.detached
    assert first.norm_types == [2.0]
    assert second.norm_types == [2.0]
    assert gradient_norm(model, math.inf) == 4.0


def test_gradient_norm_empty_model_and_invalid_norm_type():
    assert gradient_norm(FakeModel([None])) == 0.0
    with pytest.raises(ValueError, match="positive or infinity"):
        gradient_norm(FakeModel([]), 0)
    with pytest.raises(ValueError, match="positive or infinity"):
        gradient_norm(FakeModel([]), -math.inf)
    with pytest.raises(TypeError, match="parameters"):
        gradient_norm(object())


def test_torch_logger_requires_something_to_log():
    logger = TorchLogger(RecordingTracker())
    with pytest.raises(ValueError, match="requires loss"):
        logger.log()
