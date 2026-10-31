"""
Synthetic-testbed ablation with the two leaks removed:
  * local window now includes the current hour (t0-23 .. t0),
  * ARFIMA is fitted and run on the OBSERVED local series (obs_local), not on true_local.
Pass --legacy to reproduce the original (leaky) configuration for side-by-side reporting.

usage: python run_synthetic_fixed.py --seed 7 [--coupling 1.0] [--legacy]
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from data_gen import generate_testbed            # original generator, unchanged
from arfima import SimpleARFIMA
from run_real import train_predict
import common

OUT = os.path.join(HERE, 'results_rev'); os.makedirs(OUT, exist_ok=True)


def windows(d, legacy):
    n = len(d['t']); h = d['hour']
    xL = np.stack([d['obs_local'], np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), d['wind_speed']], 1)
    xN = np.zeros((5, n, 5))
    for i in range(5):
        xN[i, :, 0] = d['obs_neighbors'][i]; xN[i, :, 1] = d['conf_neighbors'][i]
        xN[i, :, 2] = np.cos(d['wind_dir'] - d['bearing'][i]); xN[i, :, 3] = d['dist'][i]; xN[i, :, 4] = d['wind_speed']
    t0 = np.arange(24, n - 1)
    Xl = np.stack([xL[t-24:t] if legacy else xL[t-23:t+1] for t in t0]).astype(np.float32)
    Xn = np.transpose(xN[:, t0, :], (1, 0, 2)).astype(np.float32)
    return Xl, Xn, d['true_local'][t0 + 1].astype(np.float32), t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--coupling', type=float, default=1.0)
    ap.add_argument('--legacy', action='store_true'); a = ap.parse_args()
    from models import build_single_node_gru, build_es_bigru_net
    d = generate_testbed(seed=42, coupling_mult=a.coupling)
    Xl, Xn, y, t0 = windows(d, a.legacy)
    tr, va, te = slice(0, 2922), slice(2922, 3548), slice(3548, len(y))
    Xl, Xn = common.normalise(dict(X_local=Xl, X_neigh=Xn), tr)
    ser = d['true_local'] if a.legacy else d['obs_local']
    arf = SimpleARFIMA(5, 80).fit(ser[:t0[tr.stop]])
    L = arf.forecast_series(ser, t0[0], t0[-1])[t0 + 1]; L = np.where(np.isnan(L), y, L)
    sp = lambda c: (lambda q: build_es_bigru_net(q, use_calibration=c))
    XS = ([Xl[tr], Xn[tr]], [Xl[va], Xn[va]], [Xl[te], Xn[te]]); XG = (Xl[tr], Xl[va], Xl[te])
    preds = {
        'persistence_obs': d['obs_local'][t0][te],
        'arfima': L[te],
        'gru': train_predict(build_single_node_gru, *XG, y[tr], y[va], a.seed)[1],
        'arfima_gru': train_predict(build_single_node_gru, *XG, (y-L)[tr], (y-L)[va], a.seed)[1] + L[te],
        'spatial_nocal': train_predict(sp(False), *XS, y[tr], y[va], a.seed)[1],
        'spatial_cal': train_predict(sp(True), *XS, y[tr], y[va], a.seed)[1],
        'sparc': train_predict(sp(True), *XS, (y-L)[tr], (y-L)[va], a.seed)[1] + L[te],
    }
    res = {k: common.metrics(y[te], v)['rmse'] for k, v in preds.items()}
    res.update(seed=a.seed, coupling=a.coupling, legacy=a.legacy)
    np.savez(os.path.join(OUT, f"synthetic_c{a.coupling}_s{a.seed}{'_legacy' if a.legacy else ''}.npz"), y_test=y[te], **preds)
    with open(os.path.join(OUT, 'synthetic_summary.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
    print(json.dumps(res))


if __name__ == '__main__':
    main()
