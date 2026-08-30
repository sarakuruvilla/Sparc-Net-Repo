import sys, time, json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import tensorflow as tf
from data_gen import generate_testbed
from features import build_windows, WIN
from arfima import SimpleARFIMA
from models import build_single_node_gru, build_cnn_bigru_hard_npi, build_es_bigru_net

def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true-y_pred)))
    rmse = float(np.sqrt(np.mean((y_true-y_pred)**2)))
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-y_true.mean())**2)
    r2 = float(1 - ss_res/ss_tot)
    return mae, rmse, r2

def prep_data(coupling_mult=1.0, data_seed=42):
    d = generate_testbed(seed=data_seed, coupling_mult=coupling_mult)
    Xl, Xn, y, idx = build_windows(d)
    n = len(y)
    n_train, n_val_end = 2922, 2922+626  # matches manuscript split sizes exactly
    tr = slice(0, n_train); va = slice(n_train, n_val_end); te = slice(n_val_end, n)

    # z-score norm using train stats only
    mu_l = Xl[tr].reshape(-1,4).mean(0); sd_l = Xl[tr].reshape(-1,4).std(0)+1e-8
    Xl_n = (Xl - mu_l)/sd_l
    mu_n = Xn[tr].reshape(-1,5).mean(0); sd_n = Xn[tr].reshape(-1,5).std(0)+1e-8
    Xn_n = (Xn - mu_n)/sd_n
    # calibration-confidence channel (index 1) is already bounded in [0,1] and is
    # consumed directly inside log(c_i+eps) by the spatial-attention module (Eq. 4),
    # so it is left unnormalized to keep that term numerically meaningful/non-negative.
    Xn_n[:, :, 1] = Xn[:, :, 1]
    mu_y = y[tr].mean(); sd_y = y[tr].std()

    hard_gate = np.zeros((len(y),1), dtype=np.float32)
    align = Xn[:,:,2]; ws = Xn[:,:,4]
    obs = Xn[:,:,0]
    mask = (align>0.5) & (ws>2.0)
    for i in range(len(y)):
        vals = obs[i][mask[i]]
        hard_gate[i,0] = vals.mean() if len(vals)>0 else 0.0
    mu_h = hard_gate[tr].mean(); sd_h = hard_gate[tr].std()+1e-8
    hard_gate_n = (hard_gate-mu_h)/sd_h

    # ARFIMA linear component fit on local true series, walk-forward forecasts idx-aligned
    true_local = d['true_local']
    arfima = SimpleARFIMA(ar_order=5, k_max=80).fit(true_local[:idx[n_train]])
    fc = arfima.forecast_series(true_local, idx[0], idx[-1])
    L_hat = fc[idx+1]   # forecast for target time (idx+1)
    L_hat = np.where(np.isnan(L_hat), y, L_hat)  # fallback

    return dict(Xl=Xl_n.astype(np.float32), Xn=Xn_n.astype(np.float32), hard=hard_gate_n.astype(np.float32),
                y=y.astype(np.float32), tr=tr, va=va, te=te, mu_y=mu_y, sd_y=sd_y, L_hat=L_hat.astype(np.float32))

def train_eval(build_fn, inputs_train, inputs_val, inputs_test, y_tr, y_va, y_te_raw, mu_y, sd_y, seed, target_key=''):
    y_tr_n = (y_tr - mu_y)/sd_y
    y_va_n = (y_va - mu_y)/sd_y
    m = build_fn(seed)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    m.fit(inputs_train, y_tr_n, validation_data=(inputs_val, y_va_n),
          epochs=30, batch_size=128, verbose=0, callbacks=[es])
    pred_n = m.predict(inputs_test, verbose=0).ravel()
    pred = pred_n*sd_y + mu_y
    mae, rmse, r2 = metrics(y_te_raw, pred)
    return mae, rmse, r2, pred

if __name__ == '__main__':
    t0=time.time()
    data = prep_data()
    tr,va,te = data['tr'], data['va'], data['te']
    y = data['y']
    print('n_train,n_val,n_test=', y[tr].shape[0], y[va].shape[0], y[te].shape[0])
    # sanity check ARFIMA test MAE
    Lmae = np.mean(np.abs(y[te]-data['L_hat'][te]))
    print('ARFIMA-only linear component test MAE:', Lmae)
    print('setup time', time.time()-t0)
