from timeseries.utils.typing import Module
from typing import Any, Literal
from pydantic import Field
from timeseries.configuration import ThreeQuantileConfiguration, PositiveInt, NonNegativeInt, Probability, Quantile
from timeseries.configuration import ForecastLoss as RNNLoss
import logging
from typing import Any, ClassVar
import torch
from torch import nn
from timeseries.register import ModelFactory
from timeseries.base import Base
from timeseries.utils.torch import Permute, get_activation


RNNKind = Literal["lstm", "gru"]


class ConfigRNN(ThreeQuantileConfiguration):
    """Recurrent encoder-decoder forecasting with GRU or LSTM."""
    past_steps: PositiveInt = Field(..., description='Number of past observations in each input window.')
    future_steps: PositiveInt = Field(..., description='Number of future observations to predict.')
    past_channels: PositiveInt = Field(..., description='Number of observed numerical input features.')
    future_channels: NonNegativeInt = Field(...,
                                            description='Number of known future numerical features; zero is allowed.')
    embs: list[PositiveInt] = Field(..., description='Cardinalities of the categorical input features.')
    cat_emb_dim: PositiveInt = Field(..., description='Embedding dimension for each categorical variable.')
    hidden_RNN: PositiveInt = Field(..., description='Hidden recurrent state dimension.')
    num_layers_RNN: PositiveInt = Field(..., description='Number of recurrent layers.')
    kind: RNNKind = Field(..., description='Recurrent cell architecture: lstm or gru.')
    kernel_size: PositiveInt = Field(..., description='Convolution or moving-average kernel width.')
    sum_emb: bool = Field(..., description='Sum categorical embeddings instead of concatenating them.')
    out_channels: PositiveInt = Field(..., description='Number of target variables.')
    activation: str = Field('torch.nn.ReLU', description='Fully qualified torch activation class name.')
    remove_last: bool = Field(False, description='Predict differences from the last observed target value.')
    dropout_rate: Probability = Field(0.1, description='Dropout rate.')
    use_bn: bool = Field(False, description='Use batch normalization in place of dropout.')
    persistence_weight: float = Field(0.0, description='Weight applied to the persistence-based loss penalty.', ge=0)
    loss_type: RNNLoss = Field('l1', description='Training loss selection.')
    quantiles: list[Quantile] | None = Field(None,
                                             description='Three increasing output quantiles, or None/empty for point forecasts.')
    n_classes: NonNegativeInt = Field(0, description='Number of classes; zero selects regression.')
    optim: str | None = Field(None, description='Torch optimizer class name; None selects Adam.')
    optim_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the optimizer.')
    scheduler_config: dict[str, Any] | None = Field(None, description='Keyword arguments for the StepLR scheduler.')





class MyBN(Module):
    def __init__(self, channels):
        super(MyBN, self).__init__()
        self.BN = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.BN(x.permute(0, 2, 1)).permute(0, 2, 1)


@ModelFactory.register(
    name='RNN',
    description='Recurrent encoder-decoder forecasting with GRU or LSTM.',
    id='rnn', domain='forecasting',
)
class RNN(Base):
    CONFIG_T: ClassVar[type[ConfigRNN]] = ConfigRNN

    def __init__(
            self,
            past_steps: int,
            future_steps: int,
            past_channels: int,
            future_channels: int,
            embs: list[int],
            cat_emb_dim: int,
            hidden_RNN: int,
            num_layers_RNN: int,
            kind: RNNKind,
            kernel_size: int,
            sum_emb: bool,
            out_channels: int,
            activation: str = 'torch.nn.ReLU',
            remove_last: bool = False,
            dropout_rate: float = 0.1,
            use_bn: bool = False,
            persistence_weight: float = 0.0,
            loss_type: RNNLoss = 'l1',
            quantiles: list[float] | None = None,
            n_classes: int = 0,
            optim: str | None = None,
            optim_config: dict[str, Any] | None = None,
            scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        """ Recurrent model with an encoder decoder structure

        Args:
            past_steps (int):  number of past datapoints used 
            future_steps (int): number of future lag to predict
            past_channels (int): number of numeric past variables, must be >0
            future_channels (int): number of future numeric variables 
            embs (List): list of the initial dimension of the categorical variables
            cat_emb_dim (int): final dimension of each categorical variable
            hidden_RNN (int): hidden size of the RNN block
            num_layers_RNN (int): number of RNN layers
            kind: recurrent cell architecture, either 'gru' or 'lstm'.
            kernel_size (int): kernel size in the encoder convolutional block
            sum_emb (bool): if true the contribution of each embedding will be summed-up otherwise stacked
            out_channels (int):  number of output channels
            activation (str, optional): activation fuction function pytorch. Default torch.nn.ReLU
            remove_last (bool, optional): if True the model learns the difference respect to the last seen point
            dropout_rate (float, optional): dropout rate in Dropout layers
            use_bn (bool, optional): if true BN layers will be added and dropouts will be removed
            persistence_weight (float):  weight controlling the divergence from persistence model. Default 0
            loss_type (str, optional): this model uses custom losses or l1 or mse. Custom losses can be linear_penalization or exponential_penalization. Default l1,
            quantiles (List[int], optional): we can use quantile loss il len(quantiles) = 0 (usually 0.1,0.5, 0.9) or L1loss in case len(quantiles)==0. Defaults to [].
            n_classes (int): number of classes (0 in regression)
            optim (str, optional): if not None it expects a pytorch optim method. Defaults to None that is mapped to Adam.
            optim_config (dict, optional): configuration for Adam optimizer. Defaults to None.
            scheduler_config (dict, optional): configuration for stepLR scheduler. Defaults to None.
        """
        quantiles = [] if quantiles is None else quantiles

        match activation:
            case 'torch.nn.SELU':
                logging.info('SELU does not require batch normalization')
                use_bn = False
        activation_type = get_activation(activation)

        match kind:
            case 'lstm':
                recurrent_type = nn.LSTM
            case 'gru':
                recurrent_type = nn.GRU
            case _:
                raise ValueError(f"Unsupported recurrent cell {kind!r}; expected 'lstm' or 'gru'.")

        super().__init__(
            past_steps=past_steps,
            future_steps=future_steps,
            past_channels=past_channels,
            future_channels=future_channels,
            out_channels=out_channels,
            quantiles=quantiles,
            loss_type=loss_type,
            persistence_weight=persistence_weight,
            n_classes=n_classes,
            optim=optim,
            optim_config=optim_config,
            scheduler_config=scheduler_config,
        )
        # self.device = get_device()
        self.num_layers_RNN = num_layers_RNN
        self.hidden_RNN = hidden_RNN
        self.embs = nn.ModuleList()
        self.sum_emb = sum_emb
        self.kind: RNNKind = kind
        self.remove_last = remove_last
        # assert out_channels==1, "Classification require only one channel"

        emb_channels = 0

        for k in embs:
            self.embs.append(nn.Embedding(k + 1, cat_emb_dim))
            emb_channels += cat_emb_dim

        if sum_emb and (emb_channels > 0):
            emb_channels = cat_emb_dim
            logging.info('Using sum')
        else:
            logging.info('Using stacked')

        self.initial_linear_encoder = nn.Sequential(nn.Linear(past_channels, 4),
                                                    activation_type(),

                                                    MyBN(4) if use_bn else nn.Dropout(dropout_rate),
                                                    nn.Linear(4, 8),
                                                    activation_type(),
                                                    MyBN(8) if use_bn else nn.Dropout(dropout_rate),
                                                    nn.Linear(8, hidden_RNN // 8))
        self.initial_linear_decoder = nn.Sequential(nn.Linear(future_channels, 4),
                                                    activation_type(),
                                                    MyBN(4) if use_bn else nn.Dropout(dropout_rate),
                                                    nn.Linear(4, 8),
                                                    activation_type(),
                                                    MyBN(8) if use_bn else nn.Dropout(dropout_rate),
                                                    nn.Linear(8, hidden_RNN // 8))

        self.conv_encoder = nn.Sequential(Permute(),
                                          nn.Conv1d(emb_channels + hidden_RNN // 8, hidden_RNN // 8, kernel_size,
                                                    stride=1, padding='same'), Permute(), nn.Dropout(0.3))

        if future_channels + emb_channels == 0:
            ## occhio che vuol dire che non ho futuro , per ora ci metto una pezza e uso hidden dell'encoder
            self.conv_decoder = nn.Sequential(Permute(),
                                              nn.Conv1d(hidden_RNN, hidden_RNN // 8, kernel_size=kernel_size, stride=1,
                                                        padding='same'), Permute())
        else:
            self.conv_decoder = nn.Sequential(Permute(), nn.Conv1d(future_channels + emb_channels, hidden_RNN // 8,
                                                                   kernel_size=kernel_size, stride=1, padding='same'),
                                              Permute())

        self.Encoder = recurrent_type(input_size=hidden_RNN // 8, hidden_size=hidden_RNN,
                                      num_layers=num_layers_RNN, batch_first=True)
        self.Decoder = recurrent_type(input_size=hidden_RNN // 8, hidden_size=hidden_RNN,
                                      num_layers=num_layers_RNN, batch_first=True)
        self.final_linear = nn.ModuleList()
        for _ in range(out_channels * self.mul):
            self.final_linear.append(nn.Sequential(nn.Linear(hidden_RNN, hidden_RNN // 2),
                                                   activation_type(),
                                                   MyBN(hidden_RNN // 2) if use_bn else nn.Dropout(dropout_rate),
                                                   nn.Linear(hidden_RNN // 2, hidden_RNN // 4),
                                                   activation_type(),
                                                   MyBN(hidden_RNN // 4) if use_bn else nn.Dropout(dropout_rate),
                                                   nn.Linear(hidden_RNN // 4, hidden_RNN // 8),
                                                   activation_type(),
                                                   MyBN(hidden_RNN // 8) if use_bn else nn.Dropout(dropout_rate),
                                                   nn.Linear(hidden_RNN // 8, 1)))

    def forward(self, batch):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x = batch['x_num_past'].to(self.device)

        if 'x_cat_future' in batch.keys():
            cat_future = batch['x_cat_future'].to(self.device)
        if 'x_cat_past' in batch.keys():
            cat_past = batch['x_cat_past'].to(self.device)
        if 'x_num_future' in batch.keys():
            x_future = batch['x_num_future'].to(self.device)
        else:
            x_future = None
        # import pdb

        if self.remove_last:
            idx_target = batch['idx_target'][0]

            x_start = x[:, -1, idx_target].unsqueeze(1)
            ##BxC
            x[:, :, idx_target] -= x_start

        tmp = [self.initial_linear_encoder(x)]

        for i in range(len(self.embs)):
            if self.sum_emb:
                if i > 0:
                    tmp_emb += self.embs[i](cat_past[:, :, i])
                else:
                    tmp_emb = self.embs[i](cat_past[:, :, i])
            else:
                tmp.append(self.embs[i](cat_past[:, :, i]))
        if self.sum_emb and (len(self.embs) > 0):
            tmp.append(tmp_emb)
        tot = torch.cat(tmp, 2)

        out, hidden = self.Encoder(self.conv_encoder(tot))

        tmp = []
        for i in range(len(self.embs)):
            if self.sum_emb:
                if i > 0:
                    tmp_emb += self.embs[i](cat_future[:, :, i])
                else:
                    tmp_emb = self.embs[i](cat_future[:, :, i])
            else:
                tmp.append(self.embs[i](cat_future[:, :, i]))
        if self.sum_emb and (len(self.embs)):
            tmp.append(tmp_emb)

        if x_future is not None and x_future.shape[-1] > 0:
            tmp.append(x_future)

        if len(tmp) > 0:
            tot = torch.cat(tmp, 2)
        else:
            tot = out[:, -1:, :].repeat(1, self.future_steps, 1)
        out, _ = self.Decoder(self.conv_decoder(tot), hidden)
        # else:
        #    out, _ = self.Decoder(self.conv_decoder(out),hidden)  
        res = []

        for j in range(len(self.final_linear)):
            res.append(self.final_linear[j](out))

        res = torch.cat(res, 2)
        ##BxLxC
        B, L, _ = res.shape
        res = res.reshape(B, L, -1, self.mul)

        if self.remove_last:
            res += x_start.unsqueeze(1)

        return res
