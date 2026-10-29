# Lazy financial market: unified library for financial analysis

The idea of this library is to analyse all the stock options available on `Yahoo Finance` and create an analysis on
them.
It started as a library on generic time series and then switched this kind of analysis.

## Background

What is a multivariate timeseries?

**Definition**   
A multivariate timeseries is a finite part of a realization from a stochastic process $\{X_t,t\in T\}$
in $\mathbb{R}^k$, i.e. $\{X_t,t\in T_0\}\subseteq\mathbb{R}^k$ with $T_0\subset T$

In this case $\mathbb{R}^k$ is the result of a concatenation of 2 space:

1. $\mathcal{C} \subset \mathbf{N}^c$ the set of categorical variables.
2. $\mathcal{W} \subset \mathbf{R}^{k-c}$ the set of numerical variables

For this reason it is possible to write $\forall t\in T_0,
X_t=c_tw_t$ with $c_t\in\mathcal{C}, w_t\in \mathcal{W}$
where $c_tw_t$ means concatenation.

***   
**Important**  
We assume that the categorical variables are always known, i.e. $\mathcal{C}$ is known $\forall t$.

***   

## Structure

This folder is composed of

1. **timeseries**: all the models that will be used for forecasting the timeseries
2. **notebook**: for all the simple testing
3. **tests**: this folder contains all the test for executing the timeseries library
4. **services**: API services for market data, indicators and metrics

## Timeseries

This folder contains all the models that have been implemented.

## Tests

This contains all the main test for implementing a new model in the regime.

## Services

This folder contains all the services that have to be used for sending all the information to the FE (front end).

## Model domains

Models are grouped by their registered `domain`:

| Domain               | Models                                                                                                                                         | Purpose                                                                     |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `forecasting`        | RNN (GRU/LSTM), LinearTS, Persistent, MyModel, TFT, TFT2, legacy TFT, Informer, CrossFormer, D3VAE, VVA, VQVAEA, GNN, ARIMA, EnsemblePredictor | Time series prediction, including multivariate and graph-based forecasting. |
| `volatility`         | SimpleGARCH                                                                                                                                    | Conditional volatility estimation.                                          |
| `filtering`          | KalmanFilter                                                                                                                                   | Signal filtering and smoothing.                                             |
| `technical_analysis` | MovingAverages, BollingerBands, RegimeDetector                                                                                                 | Trend indicators, price bands and volatility-based regimes.                 |
| `option_pricing`     | BlackScholes                                                                                                                                   | Analytical pricing of European call and put options.                         |

Use `ModelFactory.get_list_classes(domain="forecasting")` to filter the catalog.
Metrics have separate domains: `risk`, `returns`, `statistics` and `dependence`.

### Black–Scholes equation modelling

Black–Scholes provides a framework for pricing European call and put options.
The `BlackScholes` model implements the analytical solution and is available
through `ModelFactory` with identifier `blackscholes` and domain `option_pricing`.
No training is required.

```python
from timeseries.models import ModelFactory

model = ModelFactory.create("blackscholes", risk_free_rate = 0.05, volatility = 0.20)
call = model.price(spot = 100, strike = 100, time_to_maturity = 1.0)  # 10.4506
put = model.price(spot = 100, strike = 100, time_to_maturity = 1.0, option_type = "put")  # 5.5735
calls = model.price(spot = [90, 100, 110], strike = 100, time_to_maturity = 1.0)
```

`price()` accepts NumPy-broadcastable spot, strike and maturity inputs. It returns
a float for scalar inputs and a NumPy array otherwise. Inputs must be finite,
spot and strike positive, and maturity nonnegative. Negative interest rates are
supported. Expired options and zero volatility use their limiting payoffs.

Assume a non-dividend-paying underlying asset whose price follows geometric
Brownian motion under the risk-neutral measure:

$$
dS_t = rS_t\,dt + \sigma S_t\,dW_t,
$$

where $S_t$ is the asset price, $r$ is the continuously compounded risk-free
interest rate, $\sigma$ is volatility and $W_t$ is a standard Brownian motion
under that measure. For an option value $V(S,t)$, the Black–Scholes equation is

$$
\frac{\partial V}{\partial t}
+ \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}
+ rS\frac{\partial V}{\partial S} - rV = 0.
$$

The terminal conditions at maturity $T$, with strike price $K$, are
$C(S,T)=\max(S-K,0)$ for a call and $P(S,T)=\max(K-S,0)$ for a put.
For $\tau=T-t>0$ and $\sigma>0$, the analytical solutions are

$$
C(S,t) = S\Phi(d_1) - Ke^{-r\tau}\Phi(d_2),
\qquad
P(S,t) = Ke^{-r\tau}\Phi(-d_2) - S\Phi(-d_1),
$$

$$
d_1 = \frac{\ln(S/K)+(r+\sigma^2/2)\tau}{\sigma\sqrt{\tau}},
\qquad d_2 = d_1-\sigma\sqrt{\tau},
$$

where $\Phi$ is the standard normal cumulative distribution function and
$S,K>0$. Time to maturity is expressed in years; the rate and volatility use
the same annual basis, with decimal inputs (for example, `0.20` for 20%
annualized volatility). At maturity, use the payoff directly; for zero
volatility before maturity, use $C=\max(S-Ke^{-r\tau},0)$ and
$P=\max(Ke^{-r\tau}-S,0)$.

The model assumes constant interest rate and volatility, frictionless markets,
continuous trading and European exercise. It supplies an option-pricing
baseline rather than a forecast of future asset prices. Extensions for
dividends, changing volatility or American exercise require additional modelling.

## Supported datasets

- **Custom series**: univariate or multivariate pandas DataFrames with a `time`
  column, numeric targets, past/future covariates and categorical variables,
  loaded through `TimeSeries.load_signal()`.
- **Synthetic series**: periodic signals with noise and additive or multiplicative
  categorical effects, generated with `TimeSeries.generate_signal()` and `Categorical`.
- **Public forecasting benchmarks**: `electricity`, `etth1`, `etth2`, `ettm1`,
  `ettm2`, `exchange_rate`, `illness`, `traffic` and `weather`, loaded with
  `read_public_dataset(path, dataset)` from `timeseries.data_management.public_datasets`.
  Files must be downloaded and extracted under `path/all_six_datasets/` first.
- **Monash archive**: dataset download helpers and `.tsf` parsing through
  `Monash` and `convert_tsf_to_dataframe` in `timeseries.data_management.monash`.
- **Financial data**: historical ticker prices from Yahoo Finance through the
  service layer, with daily-return utilities in `timeseries.metrics.returns`.

Training uses windowed PyTorch datasets (`MyDataset`); token-based VVA models
use `VVADataset`. Legacy TFT and GNN workflows have dedicated data loaders.
The Venice tide builder (`build_venice`) is separate: `venice` is currently
excluded from the public dataset loader's accepted names.

## Model and metric factories

Public forecasting/statistical models use `ModelFactory`; metrics use an independent
`MetricFactory`. Models declare a Pydantic `CONFIG_T`; metrics use constructor
arguments directly and have no configuration classes. Both register with a decorator.
Metadata (`ModelInfo` / `MetricInfo`) stores the actual class,
identifier, name, description and application domain. Metric metadata also declares
supported `actions`.

```python
from timeseries.models import ModelFactory
from timeseries.metrics import MetricFactory

config = ModelFactory.get_config("kalman", measurement_variance = 0.05)
model = ModelFactory.create(config = config)
# Explicit overrides are validated without modifying config.
model = ModelFactory.create("kalman", config = config, process_variance = 0.001)

info = ModelFactory.get_info("d3vae")
fields = ModelFactory.get_config_param("d3vae")
risk_metrics = MetricFactory.get_list_classes(domain = "risk", actions = {"performance"})
metric_params = MetricFactory.get_config_param("sharpe")
# {"risk_free_rate": 0.0, "periods_per_year": 252}

sharpe = MetricFactory.create("sharpe", risk_free_rate = 0.0, periods_per_year = 252)
sharpe.update([0.01, -0.02, 0.03])
value = sharpe.compute()
sharpe.reset()
```

`ModelFactory.create()` passes validated configuration fields as constructor keyword arguments.
The constructor and `CONFIG_T` have matching parameter names and defaults. Unknown
fields and invalid constraints raise Pydantic validation errors. `ModelRegister`
and `MetricRegister` remain available as aliases.

`MetricFactory.get_config_param()` returns a fresh dictionary of constructor
defaults (`{}` for metrics without parameters). `MetricFactory.create(id, **params)`
passes arguments directly to the metric constructor, which validates its inputs.
Metrics do not provide `get_config()` or `CONFIG_T`.

To add a model, declare an explicit configuration and decorate the implementation:

```python
from pydantic import Field
from timeseries.registry import Configuration
from timeseries.models import ModelFactory


class ExampleConfig(Configuration):
    window: int = Field(20, gt = 0, description = "Number of past observations.")


@ModelFactory.register(
    name = "Example", description = "Example window-based model.",
    id = "example", domain = "forecasting",
)
class Example:
    CONFIG_T = ExampleConfig

    def __init__(
            self,
            window: int = 20
    ) -> None:
        self.window = window
```

`timeseries/models/__init__.py` recursively imports model modules to activate their
registration decorators, then exposes each registered model and its `CONFIG_T`
configuration. For example, `from timeseries.models import BlackScholes, ConfigBlackScholes`.
New model modules are discovered automatically without adding imports or module
routing entries to `model_register.py`. Direct factory imports initialize the same model
package on first use. To register a metric, use `MetricFactory.register` and define
its constructor arguments without a configuration class. Metric implementations
inherit from `torchmetrics.Metric` and can declare `actions={"performance"}`.

Financial metrics consume **one-dimensional decimal simple returns**, not prices:
`0.01` means +1%. `risk_free_rate` and `target_return` are **per-observation** decimal
returns; `periods_per_year` controls annualization. Sharpe and volatility use sample
standard deviation. Sortino uses the root mean square of negative target deviations
across all observations. Omega compares gains and shortfalls around its threshold.
Maximum drawdown includes initial capital and returns a positive loss fraction;
Calmar divides the annualized compounded return by that drawdown.

Streaming `update()` calls accumulate observations until `reset()`. Empty inputs at
`compute()` raise `ValueError`; undefined ratios return `NaN` and nonzero ratios
with a zero denominator return infinity. Windowed mean/variance/Sharpe use
non-overlapping windows and include the entire final partial window. The historical
`timeseries.metrics.RSI` module path is retained, although those classes are not RSI
indicators. The ARIMA-style model currently fits AR coefficients only; `q` remains a
reserved parameter and differencing supports orders zero and one.

Run the focused refactoring tests with `python -m pytest tests` in the project environment.

## Model layout and PyTorch training

All model implementations live under `timeseries/models`:

- Forecasting architectures and statistical models are directly in `models/`,
  for example `RNN.py`, `LinearTS.py` and `ARIMA.py`.
- Each model file defines its Pydantic configuration alongside the implementation:
  `from timeseries.models.RNN import RNN, ConfigRNN`.
- Architecture subdirectories contain neural components: `gnn/`, `tft/`,
  `legacy_tft/`, `crossformer/`, `informer/`, `d3vae/`, `tft2/` and `vva/`.

Shared infrastructure lives outside `models`: `timeseries/base.py` contains the
forecasting superclass, `timeseries/configuration.py` contains shared validation
types, `timeseries/utils/` contains utilities and losses, and
`timeseries/training.py` provides training with validation and checkpoints.
All training loops and orchestration live in `timeseries/training.py`, including
GNN and legacy TFT. `timeseries/workflows/` contains batch/loss adapters,
configuration helpers, inference, persistence and thin command line entry points.
Run entry points as package modules, for example
`python -m timeseries.workflows.gnn.run --help`.
Data generation, loading, temporal splits, scaling, datasets and modifiers all
live in `timeseries/data_management/`. The former `data_structure/` package has
been merged into it. Automated tests live in `tests/`.

Use `from timeseries.data_management import TimeSeries, Categorical` for data
preparation alone. For the model lifecycle (`set_model`, `train_model`, inference,
save/load), use `from timeseries.workflows import TimeSeries`. Existing imports
from `timeseries.data_structure` must be updated to one of these two APIs.

```python
from timeseries.models import ModelFactory
from timeseries.training import train_model

model = ModelFactory.create(
    "linearts", past_steps = 24, future_steps = 6, past_channels = 1,
    future_channels = 0, embs = [], cat_emb_dim = 4, kernel_size = 3,
    sum_emb = False, out_channels = 1, hidden_size = 32,
    optim_config = {"lr": 0.001},
)
history = train_model(
    model, train_loader, validation_loader,
    epochs = 20, device = "cpu", checkpoint_dir = "checkpoints",
)
```

The loop uses `zero_grad`, `backward`, `step`, evaluation mode and disabled gradients
during validation. It supports gradient clipping, epoch-based schedulers and
validation-driven ReduceLROnPlateau. Checkpoints
are ordinary PyTorch dictionaries containing `state_dict`, epoch, optimizer and
scheduler states; `best.pt` follows validation loss and `last.pt` records the final
epoch. Load model weights with:

```python
import torch

checkpoint = torch.load("checkpoints/best.pt", map_location = "cpu", weights_only = True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

`timeseries.workflows.TimeSeries.train_model()` uses the same loop and also writes `loss.csv`. It accepts
`device` for one CPU or GPU. Its former `auto_lr_find`, `devices` and `precision`
arguments have been removed; older notebook calls using those arguments must be
updated. Select the learning rate through the model's `optim_config`.

`train_model` selects the adapter for Base models, GNN and legacy TFT automatically.
GNN and legacy TFT accept an explicit `optimizer` and `scheduler`; Base models
configure those through `configure_optimizers()`. A new batch/loss contract can be
supported by passing an `adapter` implementing `TrainingAdapter.step`, without
adding another optimization loop. Validation runs after training every epoch;
losses are weighted by sample count, including the final partial batch. GNN
retains its scheduler interval of 100 epochs and legacy TFT its iterative training
every fifth epoch. Legacy entry points export their historical report and raw
weight files alongside the common best/last checkpoint dictionaries.

The focused tests retain numerical metric checks, factory validation, a real
optimization step and checkpoint round trips. Repetitive constructor and metadata
inspection tests have been removed.
