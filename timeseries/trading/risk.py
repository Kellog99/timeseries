"""Pre-trade cash/exposure limits and historical tail losses."""
import numpy as np
from pydantic import Field

from timeseries.registry import Configuration
from .models import Portfolio


class RiskLimits(
    Configuration
):
    max_position_fraction: float = Field(
        1,
        gt = 0,
        description = 'Maximum absolute market value of one position divided by account equity.'
    )
    max_gross_leverage: float = Field(
        1,
        gt = 0,
        description = 'Maximum sum of absolute position market values divided by account equity.'
    )
    max_drawdown: float = Field(
        0.2,
        gt = 0,
        le = 1,
        description = 'Fractional decline from peak equity that permits only risk reductions.'
    )
    allow_short: bool = Field(
        False,
        description = 'Allow negative position quantities, subject to broker restrictions.'
    )

    def check(
        self,
        portfolio: Portfolio,
        marks: dict[str, float],
        peak_equity: float,
        previous: Portfolio | None = None
    ) -> None:
        equity = portfolio.equity(
            marks
        )
        values = portfolio.market_values(
            marks
        )
        gross = sum(
            abs(
                v
            ) for v in values.values()
        )
        # Liquidation/reduction stays possible after a limit has been breached.
        reducing = previous is not None and all(
            s in previous.positions and abs(
                p.quantity
            ) <= abs(
                previous.positions[s].quantity
            )
            and p.quantity * previous.positions[s].quantity >= 0
            for s, p in portfolio.positions.items() if p.quantity
        ) and gross < sum(
            abs(
                v
            ) for v in previous.market_values(
                marks
            ).values()
        )
        if reducing:
            return
        if equity <= 0:
            raise ValueError(
                'Nonpositive account equity.'
            )
        if not self.allow_short and any(
                p.quantity < 0 for p in portfolio.positions.values()
        ):
            raise ValueError(
                'Short positions are disabled.'
            )
        if gross > equity * self.max_gross_leverage + 1e-8:
            raise ValueError(
                'Gross leverage limit exceeded.'
            )
        if any(
                abs(
                    v
                ) > equity * self.max_position_fraction + 1e-8 for v in values.values()
        ):
            raise ValueError(
                'Position concentration limit exceeded.'
            )
        if peak_equity > 0 and 1 - equity / peak_equity >= self.max_drawdown:
            raise ValueError(
                'Drawdown limit reached; only risk reductions are allowed.'
            )


class TailRisk(
    Configuration
):
    confidence: float = Field(
        0.95,
        gt = 0,
        lt = 1,
        description = 'Confidence level used for historical value at risk and expected shortfall.'
    )

    def compute(
        self,
        returns,
        capital: float = 1
    ) -> dict[str, float]:
        values = np.asarray(
            returns,
            dtype = float
        )
        if values.ndim != 1 or not values.size or not np.isfinite(
                values
        ).all() or (values < -1).any():
            raise ValueError(
                'Expected finite simple returns >= -1.'
            )
        if not np.isfinite(
                capital
        ) or capital <= 0:
            raise ValueError(
                'Capital must be positive and finite.'
            )
        losses = -values * capital
        # Worst (1-confidence) probability mass, including a fractional boundary observation.
        ordered = np.sort(
            losses
        )[::-1]
        mass = len(
            values
        ) * (1 - self.confidence)
        whole = int(
            mass
        )
        es = (ordered[:whole].sum() + (mass - whole) * ordered[min(
            whole,
            len(
                values
            ) - 1
        )]) / mass
        return {
            'var': float(
                np.quantile(
                    losses,
                    self.confidence
                )
            ), 'expected_shortfall': float(
                es
            )
        }


def stress_test(
    portfolio: Portfolio,
    marks: dict[str, float],
    scenarios: dict[str, dict[str, float]]
) -> dict[str, float]:
    """P&L under direct instrument-price shocks (option premiums, not underlying shocks)."""
    base = portfolio.equity(
        marks
    )
    results = {}
    for name, shocks in scenarios.items():
        stressed = dict(
            marks
        )
        for symbol, shock in shocks.items():
            if not np.isfinite(
                    shock
            ) or shock < -1:
                raise ValueError(
                    'Price shocks must be finite and >= -1.'
                )
            stressed[symbol] *= 1 + shock
        results[name] = portfolio.equity(
            stressed
        ) - base
    return results
