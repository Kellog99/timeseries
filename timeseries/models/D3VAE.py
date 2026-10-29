from typing import Any, ClassVar

import torch
from pydantic import Field
from torch import nn

from timeseries.base import Base
from timeseries.configuration import BaseLoss
from timeseries.configuration import ForecastConfiguration, PositiveInt, NonNegativeInt, Probability, Quantile
from timeseries.model_register import ModelFactory
from .d3vae.model import diffusion_generate, denoise_net, pred_net


class ConfigD3VAE(ForecastConfiguration):
    """Diffusion-based variational autoencoder for probabilistic forecasting."""
    past_channels: PositiveInt = Field(..., description = 'Number of observed numerical input features.')
    past_steps: PositiveInt = Field(..., description = 'Number of past observations in each input window.')
    future_steps: PositiveInt = Field(..., description = 'Number of future observations to predict.')
    future_channels: NonNegativeInt = Field(
        ..., description = 'Number of known future numerical features; zero is allowed.'
        )
    embs: list[PositiveInt] = Field(..., description = 'Cardinalities of the categorical input features.')
    out_channels: PositiveInt = Field(..., description = 'Number of target variables.')
    quantiles: list[Quantile] | None = Field(
        ..., description = 'Three increasing output quantiles, or None/empty for point forecasts.'
        )
    embedding_dimension: PositiveInt = Field(32, description = 'Feature embedding dimension.')
    scale: float = Field(0.1, description = 'Noise scale in the diffusion process.', ge = 0)
    hidden_size: PositiveInt = Field(64, description = 'Hidden size.')
    num_layers: PositiveInt = Field(2, description = 'Number of stacked network layers.')
    dropout_rate: Probability = Field(0.1, description = 'Dropout rate.')
    diff_steps: PositiveInt = Field(200, description = 'Number of diffusion timesteps.')
    loss_type: BaseLoss = Field('kl', description = 'Training loss selection.')
    beta_end: float = Field(0.01, description = 'Final diffusion noise variance.', ge = 0, le = 1)
    beta_schedule: str = Field('linear', description = 'Diffusion noise schedule: linear, quad, const, or jsd.')
    channel_mult: PositiveInt = Field(2, description = 'Channel multiplier across encoder scales.')
    mult: PositiveInt = Field(1, description = 'Network width multiplier.')
    num_preprocess_blocks: PositiveInt = Field(1, description = 'Number of preprocessing blocks.')
    num_preprocess_cells: PositiveInt = Field(3, description = 'Number of cells per preprocessing block.')
    num_channels_enc: PositiveInt = Field(16, description = 'Number of encoder channels.')
    arch_instance: str = Field('res_mbconv', description = 'Architecture cell specification.')
    num_latent_per_group: PositiveInt = Field(6, description = 'Number of latent variables in each group.')
    num_channels_dec: PositiveInt = Field(16, description = 'Number of decoder channels.')
    groups_per_scale: PositiveInt = Field(2, description = 'Number of latent groups per scale.')
    num_postprocess_blocks: PositiveInt = Field(1, description = 'Number of postprocessing blocks.')
    num_postprocess_cells: PositiveInt = Field(2, description = 'Number of cells per postprocessing block.')
    beta_start: float = Field(0, description = 'Initial diffusion noise variance.', ge = 0, le = 1)
    freq: str = Field('h', description = 'Time feature frequency code.')
    optim: str | None = Field(None, description = 'Torch optimizer class name; None selects Adam.')
    optim_config: dict[str, Any] | None = Field(None, description = 'Keyword arguments for the optimizer.')
    scheduler_config: dict[str, Any] | None = Field(None, description = 'Keyword arguments for the StepLR scheduler.')


def copy_parameters(
        net_source: torch.nn.Module,
        net_dest: torch.nn.Module,
        strict: bool = True,
) -> None:
    """
    Copies parameters from one network to another.
    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    strict:
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """

    net_dest.load_state_dict(net_source.state_dict(), strict = strict)


@ModelFactory.register(
    name = 'D3VAE',
    description = 'Diffusion-based variational autoencoder for probabilistic forecasting.',
    domain = 'forecasting',
)
class D3VAE(Base):
    CONFIG_T: ClassVar[type[ConfigD3VAE]] = ConfigD3VAE

    def __init__(
            self,
            past_channels: int,
            past_steps: int,
            future_steps: int,
            future_channels: int,
            embs: list[int],
            out_channels: int,
            quantiles: list[float] | None,
            embedding_dimension: int = 32,
            scale: float = 0.1,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
            diff_steps: int = 200,
            loss_type: BaseLoss = 'kl',
            beta_end: float = 0.01,
            beta_schedule: str = 'linear',
            channel_mult: int = 2,
            mult: int = 1,
            num_preprocess_blocks: int = 1,
            num_preprocess_cells: int = 3,
            num_channels_enc: int = 16,
            arch_instance: str = 'res_mbconv',
            num_latent_per_group: int = 6,
            num_channels_dec: int = 16,
            groups_per_scale: int = 2,
            num_postprocess_blocks: int = 1,
            num_postprocess_cells: int = 2,
            beta_start: float = 0,
            freq: str = 'h',
            optim: str | None = None,
            optim_config: dict[str, Any] | None = None,
            scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        quantiles = [] if quantiles is None else quantiles
        input_dim = past_channels
        sequence_length = past_steps
        prediction_length = future_steps
        target_dim = out_channels
        super().__init__(
            past_steps = past_steps,
            future_steps = future_steps,
            past_channels = past_channels,
            future_channels = future_channels,
            out_channels = out_channels,
            quantiles = quantiles,
            loss_type = loss_type,
            optim = optim,
            optim_config = optim_config,
            scheduler_config = scheduler_config,
        )

        self.gen_net = diffusion_generate(
            target_dim, embedding_dimension, prediction_length, sequence_length, scale,
            hidden_size, num_layers, dropout_rate, diff_steps, loss_type, beta_end,
            beta_schedule, channel_mult, mult,
            num_preprocess_blocks, num_preprocess_cells, num_channels_enc, arch_instance,
            num_latent_per_group, num_channels_dec, groups_per_scale,
            num_postprocess_blocks, num_postprocess_cells
            ).to(self.device)

        self.denoise_net = denoise_net(
            target_dim, embedding_dimension, prediction_length, sequence_length, scale,
            hidden_size, num_layers, dropout_rate, diff_steps, loss_type, beta_end,
            beta_schedule, channel_mult, mult,
            num_preprocess_blocks, num_preprocess_cells, num_channels_enc, arch_instance,
            num_latent_per_group, num_channels_dec, groups_per_scale, num_postprocess_blocks,
            num_postprocess_cells, beta_start, input_dim, freq, embs
            ).to(self.device)
        self.diff_step = diff_steps
        self.pred_net = pred_net(
            target_dim, embedding_dimension, prediction_length, sequence_length, scale,
            hidden_size, num_layers, dropout_rate, diff_steps, loss_type, beta_end, beta_schedule,
            channel_mult, mult,
            num_preprocess_blocks, num_preprocess_cells, num_channels_enc, arch_instance,
            num_latent_per_group, num_channels_dec, groups_per_scale, num_postprocess_blocks,
            num_postprocess_cells, beta_start, input_dim, freq, embs
            ).to(self.device)
        # self.embedding = DataEmbedding(input_dim, embedding_dimension, freq,dropout_rate)

        self.psi = 0.5
        self.gamma = 0.01
        self.lambda1 = 1.0

        self.use_quantiles = False
        self.loss = nn.MSELoss()

    def configure_optimizers(
            self
            ):
        """
        Each model has optim_config and scheduler_config
        
        :meta private:
        """
        return self._configure_optimizer(self.denoise_net.parameters())

    def training_step(
            self,
            batch,
            batch_idx: int = 0
            ):
        """
        
        :meta private:
        """
        sample, y_noisy, recon, loss2, total_c = self(batch)
        ##self.compute_loss(batch,y_hat)
        mse_loss = self.loss(sample, y_noisy)
        loss1 = - torch.mean(torch.sum(recon, dim = [1, 2, 3]))
        loss = loss1 * self.psi + loss2 * self.lambda1 + mse_loss - self.gamma * total_c

        return loss

    def validation_step(
            self,
            batch,
            batch_idx: int = 0
            ):
        """
        
        :meta private:
        """

        copy_parameters(self.denoise_net, self.pred_net)
        batch_x = batch['x_num_past'].to(self.device)
        batch_x_mark = batch['x_cat_past'].to(self.device)
        batch_y = batch['y'].to(self.device)

        _, out, _, _ = self.pred_net(batch_x, batch_x_mark)
        mse = self.loss(out.squeeze(1), batch_y)

        return mse

    def forward(
            self,
            batch: dict[str, torch.Tensor]
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = batch['x_num_past'].shape[0]

        t = torch.randint(0, self.diff_step, (B,)).long().to(self.device)

        batch_x = batch['x_num_past'].to(self.device)
        x_mark = batch['x_cat_past'].to(self.device)
        batch_y = batch['y'].to(self.device)

        output, y_noisy, total_c, _, loss2 = self.denoise_net(batch_x, x_mark, batch_y, t)
        recon = output.log_prob(y_noisy)
        sample = output.sample()

        return sample, y_noisy, recon, loss2, total_c

    def inference(
            self,
            batch: dict[str, torch.Tensor]
            ) -> torch.Tensor:
        """Care here, we need to implement it because for predicting the N-step it will use the prediction at step
        N-1. TODO fix if because I did not implement the
        know continuous variable presence here

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        copy_parameters(self.denoise_net, self.pred_net)

        batch_x = batch['x_num_past'].float().to(self.device)
        batch_x_mark = batch['x_cat_past'].to(self.device)
        _, out, _, _ = self.pred_net(batch_x, batch_x_mark)
        # import pdb
        # pdb.set_trace()
        return torch.permute(out, (0, 2, 1, 3))
