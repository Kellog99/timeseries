
import numpy as np
import pytest
from pydantic import ValidationError

from timeseries.models import ModelRegister
from timeseries.metrics import MetricRegister
from timeseries.registry import Configuration, Factory


def test_model_configurations_live_with_their_implementations():
    from timeseries.models.RNN import RNN, ConfigRNN
    from timeseries.base import Base

    assert issubclass(RNN, Base)
    assert RNN.CONFIG_T is ConfigRNN
    for class_id in ModelRegister.get_list_classes():
        info = ModelRegister.get_info(class_id)
        assert info.config_type.__module__ == info.class_type.__module__
        assert info.config_type.__name__.startswith("Config")


def test_factory_infers_config_and_revalidates_overrides_without_mutating_it():
    config = ModelRegister.get_config("kalman", process_variance=0.2)
    model = ModelRegister.create(config=config, process_variance=0.4)
    assert model.process_variance == 0.4
    assert config.process_variance == 0.2
    with pytest.raises(ValidationError):
        ModelRegister.create(config=config, measurement_variance=0)
    with pytest.raises(ValidationError):
        ModelRegister.create("kalman", typo=1)
    with pytest.raises(TypeError):
        ModelRegister.create("garch", config=config)
    with pytest.raises(ValueError):
        ModelRegister.create("missing")
    with pytest.raises(ValueError):
        ModelRegister.create()


def test_decorator_and_registry_isolation():
    class First(Factory):
        pass
    class Second(Factory):
        pass
    class Config(Configuration):
        value: int = 1
    @First.register("Example", "Test registration", domain="test")
    class Example:
        CONFIG_T = Config
        def __init__(self, value: int = 1):
            self.value = value
    assert First.create(config=Config(value=3)).value == 3
    assert not Second.get_list_classes()
    assert "example" not in ModelRegister.get_list_classes()
    with pytest.raises(ValueError, match="Duplicate"):
        First.register("Example", "Duplicate registration", domain="test")(Example)
    assert "risk" not in ModelRegister.get_list_classes()
    assert MetricRegister.get_list_classes(domain="risk")


def test_constraints_are_applied():
    with pytest.raises(ValidationError):
        ModelRegister.get_config("garch", alpha=0.5, beta=0.6)
    with pytest.raises(ValidationError):
        ModelRegister.get_config("arima", p=0)
    with pytest.raises(ValidationError):
        MetricRegister.get_config("sharpe", periods_per_year=0)
    with pytest.raises(ValidationError):
        MetricRegister.get_config("moving_average", period=-1)
    with pytest.raises(ValidationError):
        MetricRegister.get_config("omega", threshold=float("nan"))


def test_statistical_models_work_through_factory():
    arima = ModelRegister.create("arima", p=2, d=0)
    data = np.array([2.0 ** i for i in range(10)])
    arima.fit(data)
    np.testing.assert_allclose(arima.predict(data, steps=2), [1024, 2048])
    assert np.isfinite(ModelRegister.create("garch").fit_predict([.1, -.1], steps=3)).all()
    ensemble = ModelRegister.create("ensemble")
    first = ensemble.predict(np.arange(1., 40.))
    assert first == ensemble.predict(np.arange(1., 40.))


