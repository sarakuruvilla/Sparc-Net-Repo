import sys, time, json, os
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import tensorflow as tf
from data_gen import generate_testbed
from features import build_windows
from arfima import SimpleARFIMA
from models import build_single_node_gru, build_es_bigru_net
from harness import metrics, train_eval

SWEEP_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'sweep_results.json')

def prep_data_coupling(coupling_mult, data_seed=42):
    d = generate_testbed(seed=data_seed, coupling_mult=coupling_mult)
    Xl, Xn, y, idx = build_windows(d)
    n = len(y)
    n_train, n_val_end = 2922, 2922+626
    tr = slice(0, n_train); va = slice(n_train, n_val_end); te = slice(n_val_end, n)
    mu_l = Xl[tr].reshape(-1,4).mean(0); sd_l = Xl[tr].reshape(-1,4).std(0)+1e-8
    Xl_n = (Xl - mu_l)/sd_l
    mu_n = Xn[tr].reshape(-1,5).mean(0); sd_n = Xn[tr].reshape(-1,5).std(0)+1e-8
    Xn_n = (Xn - mu_n)/sd_n
    Xn_n[:, :, 1] = Xn[:, :, 1]
    mu_y = y[tr].mean(); sd_y = y[tr].std()
    true_local = d['true_local']
    arfima = SimpleARFIMA(ar_order=5, k_max=80).fit(true_local[:idx[n_train]])
    fc = arfima.forecast_series(true_local, idx[0], idx[-1])
    L_hat = fc[idx+1]
    L_hat = np.where(np.isnan(L_hat), y, L_hat)
    return dict(Xl=Xl_n.astype(np.float32), Xn=Xn_n.astype(np.float32),
                y=y.astype(np.float32), tr=tr, va=va, te=te, mu_y=mu_y, sd_y=sd_y,
                L_hat=L_hat.astype(np.float32))

def load_state():
    if os.path.exists(SWEEP_PATH):
        with open(SWEEP_PATH) as f: return json.load(f)
    return {}

def save_state(s):
    with open(SWEEP_PATH,'w') as f: json.dump(s, f, indent=2)

if __name__ == '__main__':
    coupling = float(sys.argv[1])
    seed = int(sys.argv[2])
    key = f'c{coupling}'
    state = load_state()
    if key not in state:
        state[key] = {'single_gru_rmse': [], 'sparc_rmse': [], 'seeds': []}
    if seed in state[key]['seeds']:
        print(f'coupling {coupling} seed {seed} already done'); sys.exit(0)

    t0=time.time()
    data = prep_data_coupling(coupling)
    tr,va,te = data['tr'], data['va'], data['te']
    y=data['y']; Xl=data['Xl']; Xn=data['Xn']; mu_y=data['mu_y']; sd_y=data['sd_y']; L_hat=data['L_hat']

    mae,rmse,r2,pred = train_eval(build_single_node_gru, Xl[tr], Xl[va], Xl[te], y[tr],y[va],y[te], mu_y,sd_y, seed)
    state[key]['single_gru_rmse'].append(rmse)

    resid = y - L_hat
    build_fn = lambda s: build_es_bigru_net(s, use_calibration=True)
    mae_r,rmse_r,r2_r,pred_resid_n = train_eval(build_fn, [Xl[tr],Xn[tr]], [Xl[va],Xn[va]], [Xl[te],Xn[te]],
                                                 resid[tr],resid[va],resid[te], resid[tr].mean(), resid[tr].std(), seed)
    pred_sp = pred_resid_n + L_hat[te]
    mae2,rmse2,r22 = metrics(y[te], pred_sp)
    state[key]['sparc_rmse'].append(rmse2)
    state[key]['seeds'].append(seed)

    save_state(state)
    print(f'coupling={coupling} seed={seed} done in {time.time()-t0:.1f}s | GRU_rmse={rmse:.3f} SPARC_rmse={rmse2:.3f}')
