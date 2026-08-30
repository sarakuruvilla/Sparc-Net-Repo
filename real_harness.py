import sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import tensorflow as tf
from real_data_pipeline import build_real_dataset
from real_features import build_windows_real
from arfima import SimpleARFIMA
from models import build_single_node_gru, build_es_bigru_net
from harness import metrics

def prep_real_data():
    d = build_real_dataset()
    Xl, Xn, y = build_windows_real(d)
    n = len(y)
    n_train = int(n*0.70); n_val_end = int(n*0.85)
    tr = slice(0, n_train); va = slice(n_train, n_val_end); te = slice(n_val_end, n)

    mu_l = Xl[tr].reshape(-1,4).mean(0); sd_l = Xl[tr].reshape(-1,4).std(0)+1e-8
    Xl_n = (Xl - mu_l)/sd_l
    mu_n = Xn[tr].reshape(-1,5).mean(0); sd_n = Xn[tr].reshape(-1,5).std(0)+1e-8
    Xn_n = (Xn - mu_n)/sd_n
    Xn_n[:,:,1] = Xn[:,:,1]  # confidence channel fixed at 1 (no calibration metadata), left unnormalized
    mu_y = y[tr].mean(); sd_y = y[tr].std()

    target = d['target']
    arfima = SimpleARFIMA(ar_order=5, k_max=80).fit(target[:n_train])
    fc = arfima.forecast_series(target, 24, len(target)-2)
    idx = np.arange(24, len(target)-1)
    L_hat_full = np.full(len(target), np.nan)
    L_hat_full[idx+1] = fc[idx+1]
    L_hat = L_hat_full[25:25+n]  # align with y (target starting at t0=24 -> y is target[25:])
    L_hat = np.where(np.isnan(L_hat), y, L_hat)

    print(f'n_train={n_train} n_val={n_val_end-n_train} n_test={n-n_val_end}')
    return dict(Xl=Xl_n.astype(np.float32), Xn=Xn_n.astype(np.float32), y=y.astype(np.float32),
                tr=tr, va=va, te=te, mu_y=mu_y, sd_y=sd_y, L_hat=L_hat.astype(np.float32))

if __name__ == '__main__':
    t0=time.time()
    data = prep_real_data()
    print('prep time', time.time()-t0)
    print('ARFIMA test MAE (sanity):', np.mean(np.abs(data['y'][data['te']]-data['L_hat'][data['te']])))
