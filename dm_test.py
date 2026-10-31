"""
Diebold-Mariano test (with the Harvey-Leybourne-Newbold small-sample correction) on paired
test-set forecast errors, plus summary tables across stations, horizons and seeds.

For neural models the per-sample prediction is averaged over seeds before testing
(an ensemble forecast), so the test compares forecasts rather than training runs.

usage:
  python dm_test.py                 # all stations/horizons found in results_rev/
  python dm_test.py --loss abs      # absolute-error loss instead of squared error
"""
import os, glob, argparse, re
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_rev')


def dm(e1, e2, h=1, loss='sq'):
    """H0: equal accuracy. Negative statistic => model 1 more accurate. Returns (stat, p)."""
    d = (e1 ** 2 - e2 ** 2) if loss == 'sq' else (np.abs(e1) - np.abs(e2))
    n = len(d); dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)        # HLN correction
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p)


def load_forecasts():
    rows = {}
    for f in glob.glob(os.path.join(RES, '*.npz')):
        m = re.match(r'(.+)_h(\d+)_([a-z_]+?)(?:_s(\d+))?\.npz$', os.path.basename(f))
        if not m:
            continue
        st, h, model = m.group(1), int(m.group(2)), m.group(3)
        z = np.load(f)
        rows.setdefault((st, h, model), []).append(z)
    out = {}
    for k, zs in rows.items():
        out[k] = dict(y=zs[0]['y_test'], mask=zs[0]['mask_test'],
                      pred=np.mean([z['pred_test'] for z in zs], axis=0),
                      rmse_seeds=[float(np.sqrt(np.mean((z['y_test'][z['mask_test']] - z['pred_test'][z['mask_test']]) ** 2))) for z in zs])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--loss', default='sq', choices=['sq', 'abs'])
    ap.add_argument('--ref', default='sparc'); a = ap.parse_args()
    F = load_forecasts()
    perf, tests = [], []
    for (st, h, model), v in sorted(F.items()):
        e = v['y'][v['mask']] - v['pred'][v['mask']]
        perf.append(dict(station=st, horizon=h, model=model, n_seeds=len(v['rmse_seeds']),
                         rmse_ens=np.sqrt(np.mean(e ** 2)), mae_ens=np.mean(np.abs(e)),
                         rmse_seed_mean=np.mean(v['rmse_seeds']), rmse_seed_sd=np.std(v['rmse_seeds'])))
    for (st, h, model), v in sorted(F.items()):
        if model == a.ref or (st, h, a.ref) not in F:
            continue
        r = F[(st, h, a.ref)]
        mk = r['mask']
        stat, p = dm(r['y'][mk] - r['pred'][mk], v['y'][mk] - v['pred'][mk], h=h, loss=a.loss)
        tests.append(dict(station=st, horizon=h, ref=a.ref, vs=model, dm_stat=stat, p_value=p,
                          ref_better=stat < 0 if stat == stat else None))
    P = pd.DataFrame(perf); T = pd.DataFrame(tests)
    P.to_csv(os.path.join(RES, 'performance_table.csv'), index=False)
    T.to_csv(os.path.join(RES, 'dm_tests.csv'), index=False)
    if len(P):
        print(P.pivot_table(index=['station', 'horizon'], columns='model', values='rmse_ens').round(2).to_string())
    if len(T):
        print('\nDiebold-Mariano vs', a.ref, '(negative stat = ref more accurate)')
        print(T.round(4).to_string(index=False))
        # Holm correction within each horizon
        for h, g in T.groupby('horizon'):
            k = len(g)
            ps = np.sort(g['p_value'].values)
            holm = [min(1, max((k - i) * ps[i] for i in range(j + 1))) for j in range(k)]
            print(f'h={h}: {sum(np.array(holm) < 0.05)}/{k} comparisons significant after Holm correction')


if __name__ == '__main__':
    main()
