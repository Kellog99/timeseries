from importlib import import_module

Model = import_module("timeseries.models.legacy_tft.model-checkpoint").Model


if __name__=='__main__':
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
    train_dl, _, _, _ = dataloading(batch_size=bs, batch_size_test=bs_test, 
                                                        seq_len=seq_len, lag=lag,
                                                        hour_learning=hour, 
                                                        hour_inference=hour_test, 
                                                        train_bool=train,
                                                        step = step,
                                                        scaler_y = scaler_type,
                                                        path=path_data)
    
    d_model = 4
    n_enc = 2
    n_dec = 2
    head_size = 2
    num_heads = 2
    fw_exp = 3
    device = 'cpu'
    dropout = 0.1
    num_lstm_layers = 3
    use_target_past = True
    use_yprec = True
    n_cat_var = 8 # 6 from dataloader, but delete 'year' and add 'pos_seq','pos_fut','is_fut' (-1 +3)
    n_target_var = 1

    model = Model(use_target_past,
                  use_target_past,
                  n_cat_var,
                  n_target_var,
                  seq_len,
                  lag,
                  d_model,
                  n_enc,
                  n_dec,
                  head_size,
                  num_heads,
                  fw_exp,
                  dropout,
                  num_lstm_layers,
                  device)

    x, y = next(iter(train_dl))
    # x.shape = [8, 256, 6]
    # y.shape = [8, 256]
    categorical = x[:,:,1:]
    out = model(categorical, y)

