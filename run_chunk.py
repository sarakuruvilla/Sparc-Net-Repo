import sys, time, json, os
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import tensorflow as tf
from harness import prep_data, metrics, train_eval
from models import build_single_node_gru, build_cnn_bigru_hard_npi, build_es_bigru_net

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'results_accum.json')
ERR_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'errors_accum.npz')

def load_state():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {m: {'mae': [], 'rmse': [], 'r2': [], 'seeds': []} for m in
                   ['single_gru','cnn_bigru_hard','es_bigru_no_calib','es_bigru_proposed','es_bigru_sp']}
    if os.path.exists(ERR_PATH):
        d = dict(np.load(ERR_PATH))
    else:
        d = {}
    return results, d

def save_state(results, err_dict):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(ERR_PATH, **err_dict)

def run_seed(seed, data):
    tr,va,te = data['tr'], data['va'], data['te']
    y = data['y']; Xl = data['Xl']; Xn = data['Xn']; hard = data['hard']
    mu_y, sd_y = data['mu_y'], data['sd_y']
    L_hat = data['L_hat']
    out = {}
    errs = {}

    mae,rmse,r2,pred = train_eval(build_single_node_gru, Xl[tr], Xl[va], Xl[te],
                                   y[tr], y[va], y[te], mu_y, sd_y, seed)
    out['single_gru']=(mae,rmse,r2); errs['single_gru']=np.abs(y[te]-pred)

    mae,rmse,r2,pred = train_eval(build_cnn_bigru_hard_npi, [Xl[tr],hard[tr]], [Xl[va],hard[va]], [Xl[te],hard[te]],
                                   y[tr], y[va], y[te], mu_y, sd_y, seed)
    out['cnn_bigru_hard']=(mae,rmse,r2); errs['cnn_bigru_hard']=np.abs(y[te]-pred)

    build_fn = lambda s: build_es_bigru_net(s, use_calibration=False)
    mae,rmse,r2,pred = train_eval(build_fn, [Xl[tr],Xn[tr]], [Xl[va],Xn[va]], [Xl[te],Xn[te]],
                                   y[tr], y[va], y[te], mu_y, sd_y, seed)
    out['es_bigru_no_calib']=(mae,rmse,r2); errs['es_bigru_no_calib']=np.abs(y[te]-pred)

    build_fn = lambda s: build_es_bigru_net(s, use_calibration=True)
    mae,rmse,r2,pred = train_eval(build_fn, [Xl[tr],Xn[tr]], [Xl[va],Xn[va]], [Xl[te],Xn[te]],
                                   y[tr], y[va], y[te], mu_y, sd_y, seed)
    out['es_bigru_proposed']=(mae,rmse,r2); errs['es_bigru_proposed']=np.abs(y[te]-pred)

    resid = y - L_hat
    build_fn = lambda s: build_es_bigru_net(s, use_calibration=True)
    mae_r,rmse_r,r2_r, pred_resid_n = train_eval(build_fn, [Xl[tr],Xn[tr]], [Xl[va],Xn[va]], [Xl[te],Xn[te]],
                                   resid[tr], resid[va], resid[te], resid[tr].mean(), resid[tr].std(), seed)
    pred_sp = pred_resid_n + L_hat[te]
    mae, rmse, r2 = metrics(y[te], pred_sp)
    out['es_bigru_sp']=(mae,rmse,r2); errs['es_bigru_sp']=np.abs(y[te]-pred_sp)

    return out, errs

if __name__ == '__main__':
    seeds_to_run = [int(x) for x in sys.argv[1:]]
    t0=time.time()
    data = prep_data()
    results, err_dict = load_state()
    for seed in seeds_to_run:
        if seed in results['single_gru']['seeds']:
            print(f'seed {seed} already done, skipping'); continue
        ts=time.time()
        out, errs = run_seed(seed, data)
        for m,(mae,rmse,r2) in out.items():
            results[m]['mae'].append(mae); results[m]['rmse'].append(rmse); results[m]['r2'].append(r2)
            results[m]['seeds'].append(seed)
            key = f'{m}__seed{seed}'
            err_dict[key] = errs[m]
        print(f'seed {seed} done in {time.time()-ts:.1f}s | ' + ' '.join(f'{m}={out[m][1]:.3f}' for m in out))
        save_state(results, err_dict)
    # also store y_test, L_hat_test once
    if 'y_test' not in err_dict:
        err_dict['y_test'] = data['y'][data['te']]
        err_dict['L_hat_test'] = data['L_hat'][data['te']]
        save_state(results, err_dict)
    print('TOTAL', time.time()-t0)
