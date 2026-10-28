from typing import Any
from pydantic import Field
from timeseries.configuration import BaseLoss
from timeseries.configuration import ThreeQuantileConfiguration, PositiveInt, NonNegativeInt, Probability, Quantile
from timeseries.register import ModelFactory
from typing import Any, ClassVar
from torch import  nn
import torch
from timeseries.base import Base
from typing import List
from einops import repeat
from .crossformer.cross_encoder import Encoder
from .crossformer.cross_decoder import Decoder
from .crossformer.cross_embed import DSW_embedding
from math import ceil


class ConfigCrossFormer(ThreeQuantileConfiguration):
    """Cross-dimension attention for multivariate forecasting."""
    past_steps: PositiveInt = Field(..., description='Number of past observations in each input window.')
    future_steps: PositiveInt = Field(..., description='Number of future observations to predict.')
    past_channels: PositiveInt = Field(..., description='Number of observed numerical input features.')
    future_channels: NonNegativeInt = Field(..., description='Number of known future numerical features; zero is allowed.')
    d_model: PositiveInt = Field(..., description='Dimension of the internal feature embeddings.')
    embs: list[PositiveInt] = Field(..., description='Cardinalities of the categorical input features.')
    hidden_size: PositiveInt = Field(..., description='Hidden size.')
    n_head: PositiveInt = Field(..., description='Number of attention heads.')
    seg_len: PositiveInt = Field(..., description='Number of observations in each attention segment.')
    n_layer_encoder: PositiveInt = Field(..., description='Number of encoder layers.')
    win_size: PositiveInt = Field(..., description='Window length for merging adjacent segments.')
    out_channels: PositiveInt = Field(..., description='Number of target variables.')
    factor: PositiveInt = Field(5, description='Attention sparsity or routing factor.')
    remove_last: bool = Field(False, description='Predict differences from the last observed target value.')
    persistence_weight: float = Field(0.0, description='Weight applied to the persistence-based loss penalty.', ge=0)
    loss_type: BaseLoss = Field('l1', description='Training loss selection.')
    quantiles: list[Quantile] | None = Field(None, description='Three increasing output quantiles, or None/empty for point forecasts.')
    dropout_rate: Probability = Field(0.1, description='Dropout rate.')
    optim: str | None = Field(None, description='Torch optimizer class name; None selects Adam.')
    optim_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the optimizer.')
    scheduler_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the StepLR scheduler.')







#    self, past_channels, past_steps, future_steps, seg_len, win_size = 4,
#                factor=10, d_model=512, hidden_size = 1024, n_head=8, n_layer_encoder=3, 
#                dropout=0.0, baseline = False,
  
@ModelFactory.register(
    name='CrossFormer',
    description='Cross-dimension attention for multivariate forecasting.',
    id='crossformer', domain='forecasting',
)
class CrossFormer(Base):
    CONFIG_T: ClassVar[type[ConfigCrossFormer]] = ConfigCrossFormer

    def __init__(
        self,
        past_steps: int,
        future_steps: int,
        past_channels: int,
        future_channels: int,
        d_model: int,
        embs: list[int],
        hidden_size: int,
        n_head: int,
        seg_len: int,
        n_layer_encoder: int,
        win_size: int,
        out_channels: int,
        factor: int = 5,
        remove_last: bool = False,
        persistence_weight: float = 0.0,
        loss_type: BaseLoss = 'l1',
        quantiles: list[float] | None = None,
        dropout_rate: float = 0.1,
        optim: str | None = None,
        optim_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        """CroosFormer (https://openreview.net/forum?id=vSVLM2j9eie)

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            d_model (int):  dimension of the attention model
            embs (List): list of the initial dimension of the categorical variables
            hidden_size (int): hidden size of the linear block
            n_head (int): number of heads
            seg_len (int): segment length (L_seg) see the paper for more details
            n_layer_encoder (int):  layers to use in the encoder
            win_size (int): window size for segment merg
            factor (int): num of routers in Cross-Dimension Stage of TSA (c) see the paper
            remove_last (boolean,optional): if true the model try to predic the difference respect the last observation.
            out_channels (int):  number of output channels
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): NOT USED YET
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        quantiles = [] if quantiles is None else quantiles

        super().__init__(
            past_steps=past_steps,
            future_steps=future_steps,
            past_channels=past_channels,
            future_channels=future_channels,
            out_channels=out_channels,
            quantiles=quantiles,
            loss_type=loss_type,
            persistence_weight=persistence_weight,
            optim=optim,
            optim_config=optim_config,
            scheduler_config=scheduler_config,
        )
        self.use_quantiles = False
        self.remove_last = remove_last

        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        # The padding operation to handle invisible sgemnet length
        self.pad_past_steps = ceil(1.0 *past_steps / seg_len) * seg_len
        self.pad_future_steps = ceil(1.0 * future_steps / seg_len) * seg_len
        self.past_steps_add = self.pad_past_steps - past_steps

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, past_channels, (self.pad_past_steps // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(n_layer_encoder, win_size, d_model, n_head, hidden_size, block_depth = 1, \
                                    dropout = dropout_rate,in_seg_num = (self.pad_past_steps // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, past_channels, (self.pad_future_steps // seg_len), d_model))
        self.decoder = Decoder(seg_len, n_layer_encoder + 1, d_model, n_head, hidden_size, dropout_rate, \
                                    out_seg_num = (self.pad_future_steps // seg_len), factor = factor)
        
    def forward(self, batch):
        idx_target = batch['idx_target'][0]
        x_seq = batch['x_num_past'].to(self.device)#[:,:,idx_target]
        
        if self.remove_last:
            x_start = x_seq[:,-1,:].unsqueeze(1)
            x_seq[:,:,:]-=x_start   
 
        batch_size = x_seq.shape[0]
        if (self.past_steps_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.past_steps_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        res = predict_y[:, :self.future_steps,:].unsqueeze(3)
        if self.remove_last:
            res+=x_start.unsqueeze(1)

        return res[:, :,idx_target,:]
