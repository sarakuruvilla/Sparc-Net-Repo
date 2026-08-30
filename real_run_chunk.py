import sys, time, json, os
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import tensorflow as tf
from real_harness import prep_real_data
from models import build_single_node_gru, build_es_bigru_net
from harness import metrics, train_eval

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'real_results.json')

def load_state():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f: return json.load(f)
    return {'single_gru': {'mae':[],'rmse':[],'r2':[],'seeds':[]},
            'sparc_no_calib': {'mae':[],'rmse':[],'r2':[],'seeds':[]}}

def save_state(s):
    with open(RESULTS_PATH,'w') as f: json.dump(s, f, indent=2)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    model_name = sys.argv[2]
    state = load_state()
    if seed in state[model_name]['seeds']:
        print(f'{model_name} seed {seed} already done'); sys.exit(0)

    t0=time.time()
    data = prep_real_data()
    tr,va,te = data['tr'], data['va'], data['te']
    y=data['y']; Xl=data['Xl']; Xn=data['Xn']; mu_y=data['mu_y']; sd_y=data['sd_y']; L_hat=data['L_hat']

    if model_name=='single_gru':
        mae,rmse,r2,pred = train_eval(build_single_node_gru, Xl[tr], Xl[va], Xl[te], y[tr],y[va],y[te], mu_y,sd_y, seed)
    else:
        resid = y - L_hat
        build_fn = lambda s: build_es_bigru_net(s, use_calibration=False)
        mae_r,rmse_r,r2_r,pred_resid_n = train_eval(build_fn, [Xl[tr],Xn[tr]], [Xl[va],Xn[va]], [Xl[te],Xn[te]],
                                                     resid[tr],resid[va],resid[te], resid[tr].mean(), resid[tr].std(), seed)
        pred_sp = pred_resid_n + L_hat[te]
        mae,rmse,r2 = metrics(y[te], pred_sp)

    state[model_name]['mae'].append(mae); state[model_name]['rmse'].append(rmse)
    state[model_name]['r2'].append(r2); state[model_name]['seeds'].append(seed)
    save_state(state)
    print(f'{model_name} seed={seed} done in {time.time()-t0:.1f}s | mae={mae:.3f} rmse={rmse:.3f} r2={r2:.3f}')
