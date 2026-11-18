from pydantic import BaseModel, Field


class Fundamentals(
    BaseModel
):
    market_cap: float | None = Field(
        None,
        description = 'Market capitalization in the provider-reported currency; None if unavailable.'
    )
    pe_ratio: float | None = Field(
        None,
        description = 'Trailing price-to-earnings ratio; None if unavailable.'
    )
    pb_ratio: float | None = Field(
        None,
        description = 'Price-to-book ratio; None if unavailable.'
    )
    dividend_yield: float | None = Field(
        None,
        description = 'Trailing annual dividend yield as a decimal; None if unavailable.'
    )
    eps: float | None = Field(
        None,
        description = 'Trailing earnings per share; None if unavailable.'
    )
    beta: float | None = Field(
        None,
        description = 'Provider-reported beta relative to its benchmark; None if unavailable.'
    )
    week_52_high: float | None = Field(
        None,
        description = 'Highest trading price over the last 52 weeks; None if unavailable.'
    )
    week_52_low: float | None = Field(
        None,
        description = 'Lowest trading price over the last 52 weeks; None if unavailable.'
    )
    avg_volume: float | None = Field(
        None,
        description = 'Provider-reported average trading volume; None if unavailable.'
    )
    revenue_growth: float | None = Field(
        None,
        description = 'Revenue growth rate as a decimal; None if unavailable.'
    )
    profit_margin: float | None = Field(
        None,
        description = 'Net profit margin as a decimal; None if unavailable.'
    )
    roe: float | None = Field(
        None,
        description = 'Return on equity as a decimal; None if unavailable.'
    )
    debt_to_equity: float | None = Field(
        None,
        description = 'Debt-to-equity ratio as a decimal; None if unavailable.'
    )


class ReturnDistribution(
    BaseModel
):
    bin: str = Field(
        description = 'Label identifying the return histogram interval.'
    )
    count: int = Field(
        description = 'Number of observations in the histogram interval.'
    )
    log_count: float = Field(
        ...,
        title = "logCount",
        description = 'Logarithmic count value for the histogram interval.'
    )
    exp_count: float = Field(
        ...,
        title = "expCount",
        description = 'Exponential count value for the histogram interval.'
    )


class TimeSeries(
    BaseModel
):
    date: str = Field(
        description = 'Date associated with the observed and predicted values.'
    )
    actual: float = Field(
        description = 'Observed time-series value.'
    )
    arima: float | None = Field(
        None,
        description = 'ARIMA model output for this date; None if unavailable.'
    )
    lstm: float | None = Field(
        None,
        description = 'LSTM model output for this date; None if unavailable.'
    )
    garch: float | None = Field(
        None,
        description = 'GARCH model output for this date; None if unavailable.'
    )
    ensemble: float | None = Field(
        None,
        description = 'Combined model output for this date; None if unavailable.'
    )


class PerformanceMetrics(
    BaseModel
):
    period: str = Field(
        description = 'Label identifying the performance measurement period.'
    )
    return_value: float = Field(
        ...,
        title = "return",
        description = 'Investment return over the selected period.'
    )
    volatility: float = Field(
        description = 'Return volatility over the selected period.'
    )
    sharpe_ratio: float = Field(
        ...,
        title = "sharpeRatio",
        description = 'Sharpe ratio over the selected period.'
    )
    max_drawdown: float = Field(
        ...,
        title = "maxDrawdown",
        description = 'Maximum peak-to-trough decline over the selected period.'
    )


class TechnicalSignals(
    BaseModel
):
    rsi: float = Field(
        description = 'Relative Strength Index value.'
    )
    macd: float = Field(
        description = 'Moving Average Convergence Divergence value.'
    )
    macd_signal: float = Field(
        ...,
        title = "macdSignal",
        description = 'Signal line value associated with the MACD indicator.'
    )
    bollinger_band_position: float = Field(
        ...,
        title = "bollingerBandPosition",
        description = 'Price position relative to the Bollinger Bands.'
    )
    moving_average_50: float = Field(
        ...,
        title = "movingAverage50",
        description = 'Moving-average price over 50 observations.'
    )
    moving_average_200: float = Field(
        ...,
        title = "movingAverage200",
        description = 'Moving-average price over 200 observations.'
    )


class FinancialData(
    BaseModel
):
    ticker: str = Field(
        description = 'Trading symbol of the analyzed instrument.'
    )
    company_name: str = Field(
        ...,
        title = "companyName",
        description = 'Display name of the company.'
    )
    current_price: float = Field(
        ...,
        title = "currentPrice",
        description = 'Latest reported trading price.'
    )
    price_change: float = Field(
        ...,
        title = "priceChange",
        description = 'Absolute price change from the reference price.'
    )
    price_change_percent: float = Field(
        ...,
        title = "priceChangePercent",
        description = 'Price change from the reference price, expressed as a percentage.'
    )

    fundamentals: Fundamentals = Field(
        description = 'Company fundamentals and valuation ratios.'
    )
    return_distribution: list[ReturnDistribution] = Field(
        ...,
        title = "returnDistribution",
        description = 'Histogram intervals and counts for the return distribution.'
    )
    time_series: list[TimeSeries] = Field(
        ...,
        title = "timeSeries",
        description = 'Historical observations and available model outputs by date.'
    )
    performance_metrics: list[PerformanceMetrics] = Field(
        ...,
        title = "performanceMetrics",
        description = 'Performance measurements for each reporting period.'
    )
    technical_signals: TechnicalSignals = Field(
        ...,
        title = "technicalSignals",
        description = 'Latest technical indicator values.'
    )
