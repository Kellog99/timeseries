from typing import List, Optional

from pydantic import BaseModel, Field


class Fundamentals(BaseModel):
    market_cap: float = Field(..., title="marketCap")
    pe_ratio: float = Field(..., title="peRatio")
    pb_ratio: float = Field(..., title="pbRatio")
    dividend_yield: float = Field(..., title="dividendYield")
    eps: float = Field(..., title="eps")
    beta: float = Field(..., title="beta")
    week_52_high: float = Field(..., title="week52High")
    week_52_low: float = Field(..., title="week52Low")
    avg_volume: float = Field(..., title="avgVolume")
    revenue_growth: float = Field(..., title="revenueGrowth")
    profit_margin: float = Field(..., title="profitMargin")
    roe: float = Field(..., title="roe")
    debt_to_equity: float = Field(..., title="debtToEquity")


class ReturnDistribution(BaseModel):
    bin: str
    count: int
    log_count: float = Field(..., title="logCount")
    exp_count: float = Field(..., title="expCount")


class TimeSeries(BaseModel):
    date: str
    actual: float
    arima: Optional[float] = None
    lstm: Optional[float] = None
    garch: Optional[float] = None
    ensemble: Optional[float] = None


class PerformanceMetrics(BaseModel):
    period: str
    return_value: float = Field(..., title="return")
    volatility: float
    sharpe_ratio: float = Field(..., title="sharpeRatio")
    max_drawdown: float = Field(..., title="maxDrawdown")


class TechnicalSignals(BaseModel):
    rsi: float
    macd: float
    macd_signal: float = Field(..., title="macdSignal")
    bollinger_band_position: float = Field(..., title="bollingerBandPosition")
    moving_average_50: float = Field(..., title="movingAverage50")
    moving_average_200: float = Field(..., title="movingAverage200")


class FinancialData(BaseModel):
    ticker: str
    company_name: str = Field(..., title="companyName")
    current_price: float = Field(..., title="currentPrice")
    price_change: float = Field(..., title="priceChange")
    price_change_percent: float = Field(..., title="priceChangePercent")

    fundamentals: Fundamentals
    return_distribution: List[ReturnDistribution] = Field(..., title="returnDistribution")
    time_series: List[TimeSeries] = Field(..., title="timeSeries")
    performance_metrics: List[PerformanceMetrics] = Field(..., title="performanceMetrics")
    technical_signals: TechnicalSignals = Field(..., title="technicalSignals")
