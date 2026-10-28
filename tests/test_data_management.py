"""Regression checks for loaders and dataset contracts used by type checking."""

import numpy as np
import pandas as pd
import pytest
import torch

from timeseries.data_management import Categorical, TimeSeries
from timeseries.data_management.modifiers import VVADataset
from timeseries.data_management.monash import convert_tsf_to_dataframe
from timeseries.data_management.public_datasets import read_public_dataset
from timeseries.data_management.utils import ActionEnum
from timeseries.models.d3vae.neural_operations import act


@pytest.mark.parametrize("action", ["multiplicative", ActionEnum.multiplicative])
def test_categorical_multiplier_applies_to_generated_signal(
        action
        ):
    category = Categorical("constant", 1, [1], 1, action, [2.0])
    series = TimeSeries("example")
    series.generate_signal(length = 20, categorical_variables = [category], noise_mean = 0)
    np.testing.assert_allclose(series.dataset["signal"], series.base_signal * 2)


def test_tsf_parser_returns_data_and_metadata_without_distutils(
        tmp_path
        ):
    path = tmp_path / "example.tsf"
    path.write_text(
        "@attribute series_name string\n@frequency hourly\n@horizon 2\n"
        "@missing false\n@equallength true\n@data\nexample:1,2,3\n"
    )
    frame, frequency, horizon, missing, equal = convert_tsf_to_dataframe(str(path))
    assert (frequency, horizon, missing, equal) == ("hourly", 2, False, True)
    np.testing.assert_allclose(np.asarray(frame.loc[0, "series_value"], dtype = float), [1, 2, 3])


def test_generated_signal_sums_all_additive_categories():
    categories = [Categorical(str(level), 1, [1], 1, "additive", [level]) for level in [2., 3.]]
    series = TimeSeries("example")
    series.generate_signal(length = 20, categorical_variables = categories, noise_mean = 0)
    np.testing.assert_allclose(series.dataset["signal"], series.base_signal + 5)


@pytest.mark.parametrize(
    "dataset, relative_path", [
        ("etth1", "ETT-small/ETTh1.csv"),
        ("etth2", "ETT-small/ETTh2.csv"),
        ("exchange_rate", "exchange_rate/exchange_rate.csv"),
        ("illness", "illness/national_illness.csv"),
    ]
    )
def test_public_dataset_selects_the_corresponding_csv(
        tmp_path,
        dataset,
        relative_path
        ):
    path = tmp_path / "all_six_datasets" / relative_path
    path.parent.mkdir(parents = True)
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "OT": [1., 2.]}).to_csv(path, index = False)

    frame, covariates = read_public_dataset(str(tmp_path), dataset)

    assert frame is not None
    assert covariates == []
    assert list(frame.columns) == ["time", "y"]
    assert pd.api.types.is_datetime64_any_dtype(frame["time"])


def test_token_dataset_block_length_matches_shifted_sequence():
    dataset = VVADataset([[0, 1, 2]], [[3, 4]], [[1., 2.]], [0], 3, 2, 5)
    sample = dataset[0]
    assert dataset.get_block_size() == len(sample["x_emb"]) == len(sample["y_emb"]) == 4


def test_custom_swish_backward_matches_numerical_gradient():
    values = torch.randn(4, dtype = torch.float64, requires_grad = True)
    assert torch.autograd.gradcheck(act, (values,))
