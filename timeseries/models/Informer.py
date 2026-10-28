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
from .informer.encoder import Encoder, EncoderLayer, ConvLayer
from .informer.decoder import Decoder, DecoderLayer
from .informer.attn import FullAttention, ProbAttention, AttentionLayer
from .informer.embed import DataEmbedding


class ConfigInformer(ThreeQuantileConfiguration):
    """Sparse-attention Transformer for long sequence forecasting."""
    past_steps: PositiveInt = Field(..., description='Number of past observations in each input window.')
    future_steps: PositiveInt = Field(..., description='Number of future observations to predict.')
    past_channels: PositiveInt = Field(..., description='Number of observed numerical input features.')
    future_channels: NonNegativeInt = Field(..., description='Number of known future numerical features; zero is allowed.')
    d_model: PositiveInt = Field(..., description='Dimension of the internal feature embeddings.')
    embs: list[PositiveInt] = Field(..., description='Cardinalities of the categorical input features.')
    hidden_size: PositiveInt = Field(..., description='Hidden size.')
    n_layer_encoder: PositiveInt = Field(..., description='Number of encoder layers.')
    n_layer_decoder: PositiveInt = Field(..., description='Number of decoder layers.')
    out_channels: PositiveInt = Field(..., description='Number of target variables.')
    mix: bool = Field(True, description='Mix attention heads in the Informer decoder.')
    activation: str = Field('torch.nn.ReLU', description='Fully qualified torch activation class name.')
    remove_last: bool = Field(False, description='Predict differences from the last observed target value.')
    attn: str = Field('prob', description='Attention mechanism: prob for sparse attention or full.')
    output_attention: bool = Field(False, description='Return attention weights alongside model predictions.')
    distil: bool = Field(True, description='Use convolutional distillation between encoder layers.')
    factor: PositiveInt = Field(5, description='Attention sparsity or routing factor.')
    n_head: PositiveInt = Field(1, description='Number of attention heads.')
    persistence_weight: float = Field(0.0, description='Weight applied to the persistence-based loss penalty.', ge=0)
    loss_type: BaseLoss = Field('l1', description='Training loss selection.')
    quantiles: list[Quantile] | None = Field(None, description='Three increasing output quantiles, or None/empty for point forecasts.')
    dropout_rate: Probability = Field(0.1, description='Dropout rate.')
    optim: str | None = Field(None, description='Torch optimizer class name; None selects Adam.')
    optim_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the optimizer.')
    scheduler_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the StepLR scheduler.')




@ModelFactory.register(
    name='Informer',
    description='Sparse-attention Transformer for long sequence forecasting.',
    id='informer', domain='forecasting',
)
class Informer(Base):
    CONFIG_T: ClassVar[type[ConfigInformer]] = ConfigInformer

    def __init__(
        self,
        past_steps: int,
        future_steps: int,
        past_channels: int,
        future_channels: int,
        d_model: int,
        embs: list[int],
        hidden_size: int,
        n_layer_encoder: int,
        n_layer_decoder: int,
        out_channels: int,
        mix: bool = True,
        activation: str = 'torch.nn.ReLU',
        remove_last: bool = False,
        attn: str = 'prob',
        output_attention: bool = False,
        distil: bool = True,
        factor: int = 5,
        n_head: int = 1,
        persistence_weight: float = 0.0,
        loss_type: BaseLoss = 'l1',
        quantiles: list[float] | None = None,
        dropout_rate: float = 0.1,
        optim: str | None = None,
        optim_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        """Informer

        Args:
            past_steps (int): number of past datapoints used , not used here
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            d_model (int):  dimension of the attention model
            embs (List): list of the initial dimension of the categorical variables
            hidden_size (int): hidden size of the linear block
            n_layer_encoder (int):  layers to use in the encoder
            n_layer_decoder (int):  layers to use in the decoder
            out_channels (int):  number of output channels
            mix (bool, optional): se mix attention in generative decoder. Defaults to True.
            activation (str, optional): relu or gelu. Defaults to 'relu'.
            remove_last (boolean,optional): if true the model try to predic the difference respect the last observation.
            attn (str, optional): attention used in encoder, options:[prob, full]. Defaults to 'prob'.
            output_attention (bool, optional): visualize attention, keep it False please . Defaults to False. TODO: FIX THIS
            distil (bool, optional): whether to use distilling in encoder, using this argument means not using distilling. Defaults to True.
            factor (int, optional): probsparse attn factor. Defaults to 5.
            n_head (int, optional):  heads equal in the encoder and encoder. Defaults to 1.
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
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
        self.output_attention = output_attention
        self.remove_last = remove_last
        
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
        
        self.enc_embedding = DataEmbedding(past_channels, d_model, embs, dropout_rate)
        self.dec_embedding = DataEmbedding(future_channels, d_model, embs, dropout_rate)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout_rate, output_attention=output_attention), 
                                d_model, n_head, mix=False),
                    d_model,
                    hidden_size,
                    dropout=dropout_rate,
                    activation=activation
                ) for l in range(n_layer_encoder)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(n_layer_encoder-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, n_head, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout_rate, output_attention=False), 
                                d_model, n_head, mix=False),
                    d_model,
                    hidden_size,
                    dropout=dropout_rate,
                    activation=activation,
                )
                for l in range(n_layer_decoder)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, out_channels, bias=True)

    def forward(self,batch): 
        #x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        x_enc = batch['x_num_past'].to(self.device)
        if 'x_cat_past' in batch.keys():
            x_mark_enc = batch['x_cat_past'].to(self.device)
        else:
            x_mark_enc = None
        enc_self_mask = None
        
        x_dec = batch['x_num_future'].to(self.device)

        ##FIX THIS, now all are 0!
        x_dec[:,-self.future_steps:,:] = 0 

        if 'x_cat_future' in batch.keys():
            x_mark_dec = batch['x_cat_future'].to(self.device)
        else:
            x_mark_dec = None
        dec_self_mask = None
        dec_enc_mask = None

        if self.remove_last:
            idx_target = batch['idx_target'][0]
            x_start = x_enc[:,-1,idx_target].unsqueeze(1)
            x_enc[:,:,idx_target]-=x_start   

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        
        #import pdb
        #pdb.set_trace()
        res = dec_out[:,-self.future_steps:,:].unsqueeze(3)
        if self.remove_last:
            res+=x_start.unsqueeze(1)
        
        if self.output_attention:
            return res, attns
        else:
            return  res

