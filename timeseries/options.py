"""European volatility snapshots, portfolio sensitivities and full-repricing scenarios."""
import numpy as np
from pydantic import AwareDatetime, Field, model_validator

from timeseries.instruments import OptionContract
from timeseries.models.BlackScholes import BlackScholes, Greeks
from timeseries.registry import Configuration


class OptionMarketQuote(Configuration):
    contract: OptionContract = Field(description = 'Option contract associated with the quote.')
    observed_at: AwareDatetime = Field(
        description = 'Timezone-aware retrieval time; not a guaranteed exchange quote time.'
    )
    bid: float = Field(ge = 0, description = 'Best bid premium per underlying unit.')
    ask: float = Field(gt = 0, description = 'Best ask premium per underlying unit.')
    volume: int | None = Field(None, ge = 0, description = 'Reported number of contracts traded; None if unavailable.')
    open_interest: int | None = Field(
        None, ge = 0, description = 'Reported number of outstanding contracts; None if unavailable.'
    )
    last_trade_at: AwareDatetime | None = Field(
        None, description = 'Timezone-aware last trade time; None if unavailable.'
    )

    @model_validator(mode = 'after')
    def check_spread(
        self
    ):
        if self.bid > self.ask:
            raise ValueError('Crossed option quote.')
        return self


class OptionMarket(Configuration):
    as_of: AwareDatetime = Field(description = 'Timezone-aware valuation time for the market snapshot.')
    spot: float = Field(gt = 0, description = 'Current underlying price per unit.')
    volatility: float = Field(0.2, ge = 0, description = 'Annualized underlying volatility as a decimal.')
    risk_free_rate: float = Field(0, description = 'Annual continuously compounded risk-free rate.')
    dividend_yield: float = Field(0, ge = 0, description = 'Annual continuous dividend yield.')

    def model(
        self
    ) -> BlackScholes:
        return BlackScholes(self.risk_free_rate, self.volatility, self.dividend_yield)


class VolatilityPoint(Configuration):
    strike: float = Field(gt = 0, description = 'Option strike price for this surface node.')
    maturity: float = Field(gt = 0, description = 'Remaining time to expiry in years, using ACT/365.')
    volatility: float = Field(ge = 0, description = 'Annualized implied volatility as a decimal.')


class VolatilitySurface(Configuration):
    points: list[VolatilityPoint] = Field(
        min_length = 1, description = 'Observed volatility nodes with unique strike/maturity pairs.'
    )

    @model_validator(mode = 'after')
    def unique_nodes(
        self
    ):
        keys = [(p.strike, p.maturity) for p in self.points]
        if len(keys) != len(set(keys)):
            raise ValueError('Use one quote per strike/maturity (choose call or put).')
        return self

    @classmethod
    def from_quotes(
        cls,
        quotes: list[OptionMarketQuote],
        market: OptionMarket,
        max_age_seconds: float = 60,
        max_relative_spread: float = 0.5
    ):
        if not quotes or max_age_seconds < 0 or not 0 < max_relative_spread <= 2:
            raise ValueError('Provide quotes and valid freshness/spread limits.')
        points = []
        underlying = quotes[0].contract.underlying
        for quote in quotes:
            contract = quote.contract
            if contract.exercise != 'european' or contract.underlying != underlying:
                raise ValueError('A surface requires European contracts on the same underlying.')
            age = (market.as_of - quote.observed_at).total_seconds()
            mid = (quote.bid + quote.ask) / 2
            if not 0 <= age <= max_age_seconds or (quote.ask - quote.bid) / mid > max_relative_spread:
                raise ValueError('Stale/future observation or excessive spread.')
            t = contract.maturity(market.as_of)
            iv = market.model().implied_volatility(mid, market.spot, contract.strike, t, contract.option_type)
            points.append(VolatilityPoint(strike = contract.strike, maturity = t, volatility = iv))
        return cls(points = points)

    def volatility(
        self,
        strike: float,
        maturity: float
    ) -> float:
        """Interpolate total variance across strike and maturity; no extrapolation.

        This is a diagnostic surface, not an arbitrage-free calibration.
        """
        if not np.isfinite([strike, maturity]).all() or strike <= 0 or maturity <= 0:
            raise ValueError('Strike and maturity must be positive and finite.')
        times = sorted({p.maturity for p in self.points})
        if not times[0] <= maturity <= times[-1]:
            raise ValueError('Maturity outside the observed surface.')
        left = max(t for t in times if t <= maturity)
        right = min(t for t in times if t >= maturity)
        variances = []
        for t in (left, right):
            smile = sorted((p.strike, p.volatility ** 2 * t) for p in self.points if p.maturity == t)
            strikes, variance = zip(*smile)
            if not strikes[0] <= strike <= strikes[-1]:
                raise ValueError('Strike outside the observed smiles.')
            variances.append(float(np.interp(strike, strikes, variance)))
        if variances[1] < variances[0] - 1e-12:
            raise ValueError('Decreasing total variance at the requested strike.')
        return float(np.sqrt(np.interp(maturity, [left, right], variances) / maturity))


class GreekLimits(Configuration):
    max_abs_delta: float | None = Field(
        None, gt = 0, description = 'Maximum absolute portfolio delta per underlying; None disables the limit.'
    )
    max_abs_gamma: float | None = Field(
        None, gt = 0, description = 'Maximum absolute portfolio gamma per underlying; None disables the limit.'
    )
    max_abs_vega: float | None = Field(
        None, gt = 0,
        description = 'Maximum absolute portfolio vega per underlying, per 1.0 volatility change; None disables the '
                      'limit.'
    )

    def check(
        self,
        exposures: dict[str, Greeks]
    ) -> None:
        for underlying, greeks in exposures.items():
            for name in ('delta', 'gamma', 'vega'):
                limit = getattr(self, f'max_abs_{name}')
                if limit is not None and abs(getattr(greeks, name)) > limit:
                    raise ValueError(f'{underlying}: {name} limit exceeded.')


def portfolio_greeks(
    portfolio,
    markets: dict[str, OptionMarket]
) -> dict[str, Greeks]:
    """Sum contract-multiplied greeks by underlying; stock delta is included."""
    totals: dict[str, dict[str, float]] = {}
    for p in portfolio.positions.values():
        if not p.quantity:
            continue
        instrument = p.instrument
        if isinstance(instrument, OptionContract):
            key = instrument.underlying
            if instrument.exercise != 'european':
                raise ValueError('Analytical greeks require European exercise.')
            market = markets[key]
            greeks = market.model().greeks(
                market.spot, instrument.strike,
                instrument.maturity(market.as_of), instrument.option_type
            )
        else:
            key, greeks = instrument.symbol, Greeks(delta = 1)
        values = totals.setdefault(key, Greeks().model_dump())
        for name, value in greeks.model_dump().items():
            values[name] += value * p.quantity * instrument.multiplier
    return {key: Greeks(**value) for key, value in totals.items()}


class OptionScenario(Configuration):
    spot_shocks: dict[str, float] = Field(
        default_factory = dict, description = 'Relative spot changes by underlying symbol; 0.1 represents a 10% rise.'
    )
    volatility_shifts: dict[str, float] = Field(
        default_factory = dict,
        description = 'Additive volatility changes by underlying symbol; 0.01 is one percentage point.'
    )
    days: float = Field(0, ge = 0, description = 'Calendar days elapsed in the scenario, using ACT/365 decay.')

    @model_validator(mode = 'after')
    def valid_shocks(
        self
    ):
        if any(v <= -1 for v in self.spot_shocks.values()):
            raise ValueError('Underlying shocks must exceed -1.')
        return self

    def pnl(
        self,
        portfolio,
        markets: dict[str, OptionMarket]
    ) -> float:
        """Model-to-model P&L with full European repricing, including passage of time."""
        pnl = 0.
        for p in portfolio.positions.values():
            if not p.quantity:
                continue
            instrument = p.instrument
            key = instrument.underlying if isinstance(instrument, OptionContract) else instrument.symbol
            market = markets[key]
            spot = market.spot * (1 + self.spot_shocks.get(key, 0))
            if isinstance(instrument, OptionContract):
                if instrument.exercise != 'european':
                    raise ValueError('Scenario repricing requires European exercise.')
                vol = market.volatility + self.volatility_shifts.get(key, 0)
                model = BlackScholes(market.risk_free_rate, vol, market.dividend_yield)
                maturity = instrument.maturity(market.as_of)
                base = market.model().price(market.spot, instrument.strike, maturity, instrument.option_type)
                shocked = model.price(
                    spot, instrument.strike, max(maturity - self.days / 365, 0), instrument.option_type
                )
            else:
                base, shocked = market.spot, spot
            pnl += (shocked - base) * p.quantity * instrument.multiplier
        return float(pnl)
