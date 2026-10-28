from timeseries.models.legacy_tft.embedding_nn import *

if __name__ == '__main__':
    from timeseries.data_management.tft import dataloading
    from sklearn.preprocessing import StandardScaler

    bs = 8
    bs_test = 4
    seq_len = 265
    lag = 65
    hour = 24
    hour_test = 24
    train = True
    step = 1
    scaler_type = StandardScaler()
    path_data = '/home/andrea/timeseries/data/edison/processed.pkl'
    train_dl, _, _, _ = dataloading(
        batch_size = bs, batch_size_test = bs_test,
        seq_len = seq_len, lag = lag,
        hour_learning = hour,
        hour_inference = hour_test,
        train_bool = train,
        step = step,
        scaler_y = scaler_type,
        path = path_data
        )

    x, y = next(iter(train_dl))
    # x.shape = [8, 256, 6]
    # y.shape = [8, 256]

    # tft = True
    d_model = 4
    n_enc = 2
    n_dec = 2
    head_size = 2
    num_heads = 2
    fw_exp = 3
    device = 'cpu'
    dropout = 0.1
    n_layers = 3

    import pdb

    pdb.set_trace()
    categorical = x[:, :, 1:]

    # start embedding
    emb_dims = [12 + 1, 31 + 1, 24, 7, 3]
    n_cat_var = len(emb_dims) + 3  # added by embedding_cat_variables
    n_target_var = 1
    emb_cat_var = embedding_cat_variables(seq_len, lag, d_model, emb_dims, device)
    emb_y_var = embedding_target(d_model)
    embed_x = emb_cat_var(categorical)  #
    embed_y = emb_y_var(y.unsqueeze(dim = 2)).unsqueeze(dim = 2)  #

    tot_var = n_cat_var + n_target_var

    embed_x_past = embed_x[:, :-lag, :, :]
    embed_y_past = embed_y[:, :-lag, :, :]
    embed_x_fut = embed_x[:, -lag:, :, :]
    embed_y_fut = embed_y[:, -lag:, :, :]

    # # init NN
    from timeseries.models.legacy_tft.encoder import Encoder
    from timeseries.models.legacy_tft.decoder import Decoder

    use_target_past = False
    var_sel_enc = Encoder_Var_Selection(use_target_past, n_cat_var, n_target_var, d_model, dropout, device)
    lstm_enc = Encoder_LSTM(n_layers, d_model, dropout, device)
    grn_enc = GRN(d_model, dropout)
    encoder = Encoder(n_enc, d_model, num_heads, head_size, fw_exp, dropout)

    use_yprec = False
    var_sel_dec = Decoder_Var_Selection(use_yprec, n_cat_var, n_target_var, d_model, dropout, device)
    lstm_dec = Decoder_LSTM(n_layers, d_model, dropout)
    grn_dec = GRN(d_model, dropout)
    decoder = Decoder(n_dec, d_model, num_heads, head_size, fw_exp, lag, dropout)

    var_sel_past = var_sel_enc(embed_x_past)
    lstm_encs, hn, cn = lstm_enc(var_sel_past)
    pre_enc = grn_enc(lstm_encs)
    encoding = encoder(embed_y_past.squeeze(2), pre_enc, pre_enc)

    # DECODER USING ONLY x (prec variable to handle the difference)
    var_sel_fut = var_sel_dec(embed_x_fut)
    lstm_decs = lstm_dec(var_sel_fut, hn, cn)
    pre_dec = grn_dec(lstm_decs)
    decoding = decoder(pre_dec, encoding, encoding)

    post_transformer = postTransformer(d_model, dropout)
    out = post_transformer(decoding, pre_dec, lstm_decs)

    quantile = True
    if quantile:
        quantiles = [0.1, 0.5, 0.9]
        loss = [0] * len(quantiles)
        target = torch.randint(-2, 2, (bs, lag))
        # last_linear in data.py
        out_linear = nn.Linear(d_model, len(quantiles))
        out = out_linear(out)
        for i, q in enumerate(quantiles):
            q_loss = torch.max(q * (target - out[:, :, i]), (1 - q) * (out[:, :, i] - target))
            loss[i] = loss[i] + q_loss
