from pydantic import AwareDatetime, Field

from timeseries.registry import Configuration
from timeseries.trading import Portfolio


class PortfolioValuation(
    Configuration
):
    as_of: AwareDatetime = Field(
        description = 'Timezone-aware portfolio valuation time.'
    )
    currency: str = Field(
        description = 'Account currency used for all valuation amounts.'
    )
    cash: float = Field(
        description = 'Cash balance, including any borrowing as a negative amount.'
    )
    equity: float = Field(
        description = 'Cash balance plus the signed market value of all positions.'
    )
    gross_exposure: float = Field(
        description = 'Sum of absolute position market values.'
    )
    realized_pnl: float = Field(
        description = 'Realized profit or loss after commissions and financing costs.'
    )
    unrealized_pnl: float = Field(
        description = 'Profit or loss on open positions relative to their average entry prices.'
    )


class PortfolioSnapshot(
    Configuration
):
    as_of: AwareDatetime = Field(
        description = 'Timezone-aware time associated with the portfolio snapshot.'
    )
    portfolio: Portfolio = Field(
        description = 'Portfolio cash balances, positions and execution ledger.'
    )
    marks: dict[str, float] = Field(
        description = 'Current prices by instrument symbol, before applying contract multipliers.'
    )

    def evaluate(
        self
    ) -> PortfolioValuation:
        p = self.portfolio
        return PortfolioValuation(
            as_of = self.as_of,
            currency = p.currency,
            cash = p.cash,
            equity = p.equity(
                self.marks
            ),
            gross_exposure = sum(
                abs(
                    v
                ) for v in p.market_values(
                    self.marks
                ).values()
            ),
            realized_pnl = p.realized_pnl,
            unrealized_pnl = p.unrealized_pnl(
                self.marks
            )
        )
