from typing import Any, ClassVar

from pydantic import Field

from timeseries.base import Base
from timeseries.configuration import BaseLoss
from timeseries.configuration import ForecastConfiguration, PositiveInt
from timeseries.model_register import ModelFactory


class ConfigPersistent(ForecastConfiguration):
    """Baseline repeating the last observed target."""
    future_steps: PositiveInt = Field(..., description = 'Number of future observations to predict.')
    past_steps: PositiveInt = Field(..., description = 'Number of past observations in each input window.')
    loss_type: BaseLoss | None = Field(None, description = 'Training loss selection.')
    persistence_weight: float = Field(
        0.1, description = 'Weight applied to the persistence-based loss penalty.', ge = 0
        )
    optim_config: dict[str, Any] | None = Field(None, description = 'Keyword arguments for the optimizer.')
    scheduler_config: dict[str, Any] | None = Field(None, description = 'Keyword arguments for the StepLR scheduler.')


@ModelFactory.register(
    name = 'Persistent',
    description = 'Baseline repeating the last observed target.',
    domain = 'forecasting',
)
class Persistent(Base):
    CONFIG_T: ClassVar[type[ConfigPersistent]] = ConfigPersistent

    def __init__(
            self,
            future_steps: int,
            past_steps: int,
            loss_type: BaseLoss | None = None,
            persistence_weight: float = 0.1,
            optim_config: dict[str, Any] | None = None,
            scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        """Persistent model propagatinng  last observed values

        Args:
          
            future_steps (int): number of future lag to predict   
            past_steps (int): number of future lag to predict. Useless but needed for the other stuff

            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None. Usless for this model
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None. Usless for this
            model
        """

        super().__init__(
            past_steps = past_steps,
            future_steps = future_steps,
            loss_type = loss_type,
            persistence_weight = persistence_weight,
            optim_config = optim_config,
            scheduler_config = scheduler_config,
        )

    def forward(
            self,
            batch
            ):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        x_start = x[:, -1, idx_target].unsqueeze(1)
        # this is B,1,C

        # [B,L,C,1] remember the outoput size
        res = x_start.repeat(1, self.future_steps, 1).unsqueeze(3)

        return res
