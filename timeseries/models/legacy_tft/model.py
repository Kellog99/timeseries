import copy
import pickle as pkl
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field
from torch import nn
from tqdm import tqdm

from timeseries.configuration import ForecastConfiguration, PositiveInt, NonNegativeInt, Probability, Quantile
from timeseries.models.legacy_tft.decoder import Decoder
from timeseries.models.legacy_tft.embedding_nn import (
    Decoder_LSTM, Decoder_Var_Selection, Encoder_LSTM, Encoder_Var_Selection, GRN,
    embedding_cat_variables, embedding_target, postTransformer,
)
from timeseries.models.legacy_tft.encoder import Encoder
from timeseries.model_register import ModelFactory
from timeseries.utils.typing import Module


class ConfigLegacyTFT(ForecastConfiguration):
    """Legacy Temporal Fusion Transformer."""
    use_target_past: bool = Field(..., description = 'Include past target observations as encoder inputs.')
    use_yprec: bool = Field(..., description = 'Include previous target observations as decoder inputs.')
    n_cat_var: NonNegativeInt = Field(..., description = 'Number of categorical input variables.')
    n_target_var: PositiveInt = Field(..., description = 'Number of target variables.')
    seq_len: PositiveInt = Field(..., description = 'Total input sequence length.')
    lag: PositiveInt = Field(..., description = 'Forecast horizon length.')
    d_model: PositiveInt = Field(..., description = 'Dimension of the internal feature embeddings.')
    n_enc: PositiveInt = Field(..., description = 'Number of encoder layers.')
    n_dec: PositiveInt = Field(..., description = 'Number of decoder layers.')
    head_size: PositiveInt = Field(..., description = 'Feature dimension of each attention head.')
    num_heads: PositiveInt = Field(..., description = 'Number of attention heads.')
    fw_exp: PositiveInt = Field(..., description = 'Feed-forward hidden dimension expansion factor.')
    dropout: Probability = Field(..., description = 'Dropout.')
    num_lstm_layers: PositiveInt = Field(..., description = 'Number of LSTM layers.')
    device: str = Field(..., description = 'Torch device, for example cpu or cuda:0.')
    quantiles: list[Quantile] | None = Field(
        None, description = 'Increasing output quantiles, or None/empty for point forecasts.'
        )
    path_model_save: str | None = Field(None, description = 'Optional output path for model checkpoints.')
    model_name: str | None = Field(None, description = 'Optional model name used in saved reports.')


@ModelFactory.register(
    name = 'LegacyTFT',
    description = 'Legacy Temporal Fusion Transformer.',
    id = 'legacytft', domain = 'forecasting',
)
class Model(Module):
    CONFIG_T: ClassVar[type[ConfigLegacyTFT]] = ConfigLegacyTFT

    def __init__(
            self,
            use_target_past: bool,
            use_yprec: bool,
            n_cat_var: int,
            n_target_var: int,
            seq_len: int,
            lag: int,
            d_model: int,
            n_enc: int,
            n_dec: int,
            head_size: int,
            num_heads: int,
            fw_exp: int,
            dropout: float,
            num_lstm_layers: int,
            device: str,
            quantiles: list[float] | None = None,
            path_model_save: str | None = None,
            model_name: str | None = None,
            ) -> None:
        super().__init__()
        self.path_model_save = path_model_save
        self.model_name = model_name
        self.device = device
        self.use_target_past = use_target_past
        self.use_yprec = use_yprec
        self.lag = lag
        self.quantiles = quantiles
        # Start Embedding
        emb_dims = [12 + 1, 31 + 1, 24, 7, 3]
        self.emb_cat_var = embedding_cat_variables(seq_len, lag, d_model, emb_dims, device)
        self.emb_y_var = embedding_target(d_model)
        # Encoder (past)
        self.EncVariableSelection = Encoder_Var_Selection(
            use_target_past, n_cat_var + 3, n_target_var, d_model, dropout, device
            )
        self.EncLSTM = Encoder_LSTM(num_lstm_layers, d_model, dropout, device)
        self.EncGRN = GRN(d_model, dropout)
        self.Encoder = Encoder(n_enc, d_model, num_heads, head_size, fw_exp, dropout)
        # Decoder (future)
        self.DecVariableSelection = Decoder_Var_Selection(
            use_yprec, n_cat_var + 3, n_target_var, d_model, dropout, device
            )
        self.DecLSTM = Decoder_LSTM(num_lstm_layers, d_model, dropout)
        self.DecGRN = GRN(d_model, dropout)
        self.Decoder = Decoder(n_dec, d_model, num_heads, head_size, fw_exp, lag, dropout)
        # PostTransformer (future)
        self.postTransformer = postTransformer(d_model, dropout)
        # last linear wrt quantile
        if quantiles is not None:
            self.outLinear = nn.Linear(d_model, len(quantiles))
        else:
            self.outLinear = nn.Linear(d_model, n_target_var)

    def forward(
            self,
            categorical,
            y
            ):
        embed_categorical = self.emb_cat_var(categorical)
        embed_categorical_past = embed_categorical[:, :-self.lag, :, :]
        embed_categorical_fut = embed_categorical[:, -self.lag:, :, :]
        y_past = y[:, :-self.lag]
        embed_y_past = self.emb_y_var(y_past.unsqueeze(dim = 2))

        if self.use_target_past:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past, embed_y_past.unsqueeze(2))
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(encoding, encoding, encoding)
        else:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(embed_y_past, encoding, encoding)

        if self.use_yprec:
            ### Iterative Computation
            # Decoder_out stores y_0 and all predicted fututre values of y overwritten during for  
            decoder_out = y[:, -self.lag - 1:-1].unsqueeze(2)
            embed_y = self.emb_y_var(decoder_out)
            variable_selection_fut = self.DecVariableSelection(embed_categorical_fut, embed_y.unsqueeze(2))
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            pred_decoding = self.DecGRN(fut_LSTM)
            pred_decoded = self.Decoder(pred_decoding, encoded, encoded)
            out = self.postTransformer(pred_decoded, pred_decoding, fut_LSTM)
            out = self.outLinear(out)
        else:
            ### Direct Computation
            variable_selection_fut = self.DecVariableSelection(embed_categorical_fut)
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            decoding = self.DecGRN(fut_LSTM)
            decoded = self.Decoder(decoding, encoded, encoded)
            out = self.postTransformer(decoded, decoding, fut_LSTM)
            out = self.outLinear(out)
        return out

    def iter_forward(
            self,
            categorical,
            y
            ):
        embed_categorical = self.emb_cat_var(categorical)
        embed_categorical_past = embed_categorical[:, :-self.lag, :, :]
        embed_categorical_fut = embed_categorical[:, -self.lag:, :, :]
        y_past = y[:, :-self.lag]
        embed_y_past = self.emb_y_var(y_past.unsqueeze(dim = 2))

        if self.use_target_past:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past, embed_y_past.unsqueeze(2))
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(encoding, encoding, encoding)
        else:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(embed_y_past, encoding, encoding)

        ### Iterative Computation 
        decoder_out = y[:, -self.lag - 1].unsqueeze(1).unsqueeze(2)
        for tau in range(1, self.lag + 1):
            embed_tau_y = self.emb_y_var(decoder_out)
            variable_selection_fut = self.DecVariableSelection(
                embed_categorical_fut[:, :tau, :, :], embed_tau_y.unsqueeze(2)
                )
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            pred_decoding = self.DecGRN(fut_LSTM)
            pred_decoded = self.Decoder(pred_decoding, encoded, encoded)
            out = self.postTransformer(pred_decoded, pred_decoding, fut_LSTM)
            out = self.outLinear(out)
            decoder_out = torch.cat((decoder_out, out[:, -1, :].unsqueeze(1)), dim = 1)
        # extract only predicted y without y_0
        out = decoder_out[:, 1:, :]
        return out

    def learning(
            self,
            epochs,
            dl_training,
            dl_validation,
            cost_func,
            optimizer,
            scheduler
            ):
        """Compatibility method delegating to the shared training workflow."""
        from timeseries.training import train_legacy_tft

        return train_legacy_tft(self, epochs, dl_training, dl_validation, cost_func, optimizer, scheduler)

    def inference(
            self,
            dl,
            scaler
            ):
        if self.path_model_save is None:
            raise ValueError("A checkpoint path is required for inference.")

        self.eval()
        plt.rcParams.update({'figure.facecolor': 'lightgray'})
        fig, axs = plt.subplots(3, 2, figsize = (18, 18))
        fig.suptitle(self.model_name or type(self).__name__, fontsize = 25, fontweight = 'bold')

        # assign figure's slot
        plots = axs.flat

        loss_plot = plots[0]
        dict_loss = self.get_losses(self.path_model_save + '.pkl')
        total_epochs = dict_loss['total_epochs']
        x = torch.arange(1, dict_loss['total_epochs'])

        loss_plot.cla()
        loss_plot.set_yscale('log')
        loss_plot.plot(x, dict_loss['train'][0], label = 'train_loss')
        loss_plot.plot(x, dict_loss['val'][0], label = 'val_loss')
        loss_plot.grid(True)
        loss_plot.set_title(
            f'Epochs: {total_epochs},\n' \
            f'TRAIN: -last: {dict_loss["train"][0][-1]:.03f} -argmin: {dict_loss["train"][1]:.4f} -min: '
            f'{dict_loss["train"][2]:.4f}\n' \
            f'VAL:   -last: {dict_loss["val"][0][-1]:.03f} -argmin: {dict_loss["val"][1]:.4f} -min: '
            f'{dict_loss["val"][2]:.4f}'
            )
        loss_plot.legend()

        # * RMSE AND TEST BATCH
        net_best = self
        net_best.eval()
        net_last = copy.deepcopy(self)
        net_last.eval()

        path_best = self.path_model_save + '_best.pt'
        net_best.load_state_dict(torch.load(path_best, map_location = self.device))
        net_best.eval()

        path_last = self.path_model_save + '_last.pt'
        net_last.load_state_dict(torch.load(path_last, map_location = self.device))
        net_last.eval()

        y_data = []
        pred_best = []
        pred_last = []

        for i, (ds, y) in enumerate(tqdm(dl, desc = f" > On Test DataLoader - ")):
            ds = ds.to(self.device)
            y = y.to(self.device)
            y_clone_best = y.detach().clone().to(self.device)
            y_clone_last = y.detach().clone().to(self.device)

            output_best = net_best.iter_forward(ds[:, :, 1:], y_clone_best).detach()
            output_last = net_last.iter_forward(ds[:, :, 1:], y_clone_last).detach()

            # RESCALE THE OUTPUTS TO STORE Y_DATA,PRED_BEST, PRED_LAST
            y = scaler.inverse_transform(y[:, -self.lag:].cpu()).flatten()
            prediction_best = scaler.inverse_transform(output_best[:, -self.lag:].squeeze(2).cpu()).flatten()
            prediction_last = scaler.inverse_transform(output_last[:, -self.lag:].squeeze(2).cpu()).flatten()

            y_data.append(y.reshape(-1, self.lag))
            pred_best.append(prediction_best.reshape(-1, self.lag))
            pred_last.append(prediction_last.reshape(-1, self.lag))

            del y, y_clone_best, y_clone_last, ds
            torch.cuda.empty_cache()

        assert len(y_data) == len(pred_best)
        assert len(y_data) == len(pred_last)

        y_data = np.vstack(y_data)
        pred_best = np.vstack(pred_best)
        pred_last = np.vstack(pred_last)
        # * -- RMSE --
        rmse_best = np.sqrt(np.mean((y_data - pred_best) ** 2, axis = 0))
        rmse_last = np.sqrt(np.mean((y_data - pred_last) ** 2, axis = 0))

        mean_rmse_best = np.mean(rmse_best)
        mean_rmse_last = np.mean(rmse_last)
        min_rmse_best = rmse_best.min()
        min_rmse_last = rmse_last.min()
        max_rmse_best = rmse_best.max()
        max_rmse_last = rmse_last.max()
        x = np.arange(1, self.lag + 1)

        rmse_plot = plots[1]
        rmse_plot.cla()
        rmse_plot.plot(x, rmse_best, label = 'rmse_best')
        rmse_plot.plot(x, rmse_last, label = 'rmse_last')
        rmse_plot.grid(True)
        rmse_plot.set_title(
            f' BEST: min {min_rmse_best:.2f}, mean {mean_rmse_best:.2f}, max {max_rmse_best:.2f}\n \
                            LAST: min {min_rmse_last:.2f}, mean {mean_rmse_last:.2f}, max {max_rmse_last:.2f}'
            )
        rmse_plot.legend()

        batch_plot = plots[2:4]
        shift_plot = plots[4:]

        for k, ax in enumerate(batch_plot):
            ax.cla()
            y_lim = [0, 6000]
            ax.plot(x, pred_best[k * -1], 'p', label = 'yhat_best')
            ax.plot(x, pred_last[k * -1], 'p', label = 'yhat_last')
            ax.plot(x, y_data[k * -1], label = ' y')
            ax.set_ylim(y_lim)
            ax.grid(True)
            ax.legend()

        shift_10, shift_30 = shift_plot

        shift_10.cla()
        y_lim = [0, 6000]
        shift_10.plot(pred_best[:, 9], label = 'yhat_best_10')
        shift_10.plot(pred_last[:, 9], label = 'yhat_last_10')
        shift_10.plot(y_data[:, 9], label = ' y_10')
        shift_10.set_ylim(y_lim)
        shift_10.set_title('SHIFT 10 - ALL VALUES')
        shift_10.grid(True)
        shift_10.legend()

        shift_30.cla()
        y_lim = [0, 6000]
        shift_30.plot(pred_best[:, 29], label = 'yhat_best_30')
        shift_30.plot(pred_last[:, 29], label = 'yhat_last_30')
        shift_30.plot(y_data[:, 29], label = ' y_30')
        shift_30.set_ylim(y_lim)
        shift_30.set_title('SHIFT 30 - ALL VALUES')
        shift_30.grid(True)
        shift_30.legend()

        fig.savefig(self.path_model_save + f'_plot_{total_epochs}.png')  # ! PATH

    def get_losses(
            self,
            path
            ):
        with open(path, 'rb') as f:
            _, losses = pkl.load(f)
            f.close()
        # import pdb
        # pdb.set_trace()
        train_loss = losses['train']
        val_loss = losses['val']
        total_epochs = len(train_loss) + 1

        dict_loss = {
            # [ list, argmin, min ]
            'train': [train_loss, train_loss.index(min(train_loss)) + 1, min(train_loss)],
            'val': [val_loss, val_loss.index(min(val_loss)) + 1, min(val_loss)],
            'total_epochs': total_epochs
        }
        return dict_loss

    def count_parameters(
            self
            ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # import pdb
    # pdb.set_trace()


# Retained for existing serialized workflow configurations.
