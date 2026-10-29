import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from timeseries.metrics import MetricRegister
from timeseries.models import ModelRegister
from timeseries.registry import Configuration, Factory


@pytest.mark.parametrize('entrypoint', ['timeseries.models', 'timeseries.model_register'])
def test_model_package_initializes_and_exports_catalog_in_fresh_process(entrypoint):
    result = subprocess.run(
        [sys.executable, '-B', '-c', f'''
import importlib

entrypoint = importlib.import_module({entrypoint!r})
factory = entrypoint.ModelFactory
catalog = factory.get_list_classes()
assert {{'blackscholes', 'gnn', 'legacytft', 'rnn', 'arima'}} <= set(catalog)
models = importlib.import_module('timeseries.models')
for class_id in catalog:
    info = factory.get_info(class_id)
    assert getattr(models, info.class_type.__name__) is info.class_type
    assert getattr(models, info.config_type.__name__) is info.config_type
assert factory.get_list_classes() == catalog
assert type(factory.create(config=models.ConfigBlackScholes())) is models.BlackScholes
'''],
        cwd = Path(__file__).resolve().parents[1],
        capture_output = True, text = True, timeout = 60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    config = ModelRegister.get_config("kalman", process_variance = 0.2)
    model = ModelRegister.create(config = config, process_variance = 0.4)
    assert model.process_variance == 0.4
    assert config.process_variance == 0.2
    with pytest.raises(ValidationError):
        ModelRegister.create(config = config, measurement_variance = 0)
    with pytest.raises(ValidationError):
        ModelRegister.create("kalman", typo = 1)
    with pytest.raises(TypeError):
        ModelRegister.create("garch", config = config)
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

    @First.register("Example", "Test registration", domain = "forecasting")
    class Example:
        CONFIG_T = Config

        def __init__(
                self,
                value: int = 1
                ):
            self.value = value

    assert First.create(config = Config(value = 3)).value == 3
    assert not Second.get_list_classes()
    assert "example" not in ModelRegister.get_list_classes()
    with pytest.raises(ValueError, match = "Duplicate"):
        First.register("Example", "Duplicate registration", domain = "forecasting")(Example)
    assert "risk" not in ModelRegister.get_list_classes()
    assert MetricRegister.get_list_classes(domain = "risk")


@pytest.mark.parametrize(
    ('class_name', 'expected'), [
        ('ExampleModel', 'example'), ('EXAMPLEMODEL', 'example'),
        ('Example', 'example'), ('ModelExample', 'modelexample'),
        ('ExampleModelModel', 'examplemodel'),
    ]
    )
def test_default_id_uses_class_name_and_removes_only_trailing_model(
        class_name,
        expected
        ):
    class Registry(Factory):
        pass

    class Config(Configuration):
        pass

    implementation = type(class_name, (), {'CONFIG_T': Config})
    Registry.register('Different display name', 'Example implementation', domain = 'forecasting')(implementation)
    assert Registry.get_list_classes() == [expected]
    assert type(Registry.create(expected)) is implementation


def test_explicit_id_overrides_default_and_empty_id_is_rejected():
    class Registry(Factory):
        pass

    class Config(Configuration):
        pass

    @Registry.register('Example', 'Custom identifier', id = 'CUSTOM', domain = 'forecasting')
    class ExampleModel:
        CONFIG_T = Config

    assert Registry.get_info('custom').class_type is ExampleModel
    with pytest.raises(ValidationError):
        Registry.register('Empty', 'Invalid identifier', id = '', domain = 'forecasting')(ExampleModel)


def test_domains_validate_metadata_and_filter_registries():
    info = ModelRegister.get_info('regimedetector')
    assert info.domain == 'technical_analysis'
    assert 'regimedetector' in ModelRegister.get_list_classes(domain = {'technical_analysis', 'filtering'})
    assert 'regimedetector' not in ModelRegister.get_list_classes(domain = 'forecasting')
    with pytest.raises(ValidationError):
        type(info).model_validate(info.model_dump() | {'domain': 'unknown'})
    assert ModelRegister.get_info('my').class_type.__name__ == 'MyModel'


def test_constraints_are_applied():
    with pytest.raises(ValidationError):
        ModelRegister.get_config("garch", alpha = 0.5, beta = 0.6)
    with pytest.raises(ValidationError):
        ModelRegister.get_config("arima", p = 0)
    with pytest.raises(ValueError):
        MetricRegister.create("sharpe", periods_per_year = 0)
    with pytest.raises(ValueError):
        MetricRegister.create("moving_average", period = -1)
    with pytest.raises(ValueError):
        MetricRegister.create("omega", threshold = float("nan"))


def test_statistical_models_work_through_factory():
    arima = ModelRegister.create("arima", p = 2, d = 0)
    data = np.array([2.0 ** i for i in range(10)])
    arima.fit(data)
    np.testing.assert_allclose(arima.predict(data, steps = 2), [1024, 2048])
    assert np.isfinite(ModelRegister.create("garch").fit_predict([.1, -.1], steps = 3)).all()
    ensemble = ModelRegister.create("ensemble")
    first = ensemble.predict(np.arange(1., 40.))
    assert first == ensemble.predict(np.arange(1., 40.))
