"""
Prediction intervals on the FINAL forecast (not the linear-stage residual).
Error distributions are fitted on VALIDATION errors and evaluated on the TEST set, so no
test information is used to size the intervals. Gaussian, t-location-scale and
split-conformal intervals are compared with PICP, ACE, PINAW and the Winkler score.

usage:  python intervals.py --station Dongsi --horizon 1 --model sparc
"""
import os, glob, argparse
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_rev')
LEVELS = [0.60, 0.70, 0.80, 0.90, 0.95]


def evaluate(y, lo, hi, alpha, rng):
    cover = (y >= lo) & (y <= hi)
    picp = cover.mean(); width = hi - lo
    wink = width + (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return dict(PICP=100 * picp, ACE=100 * (picp - (1 - alpha)), PINAW=width.mean() / rng, Winkler=wink.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--station', default='Dongsi'); ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--model', default='sparc'); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RES, f'{a.station}_h{a.horizon}_{a.model}*.npz')))
    if not files:
        raise SystemExit('no prediction files found; run run_real.py first')
    zs = [np.load(f) for f in files]
    mv, mt = zs[0]['mask_val'], zs[0]['mask_test']
    ev = zs[0]['y_val'][mv] - np.mean([z['pred_val'] for z in zs], 0)[mv]
    pt = np.mean([z['pred_test'] for z in zs], 0)[mt]; yt = zs[0]['y_test'][mt]
    rng = yt.max() - yt.min()
    mu_g, sd_g = stats.norm.fit(ev)
    df_t, mu_t, sc_t = stats.t.fit(ev)
    print(f'{a.station} h={a.horizon} {a.model}: {len(files)} seed(s); val errors kurtosis={stats.kurtosis(ev):.2f}, t df={df_t:.2f}')
    print(f"{'method':<12}{'level':>6}{'PICP':>8}{'ACE':>8}{'PINAW':>8}{'Winkler':>10}")
    for lev in LEVELS:
        al = 1 - lev
        for name, (ql, qh) in {
            'gaussian': stats.norm.ppf([al / 2, 1 - al / 2], mu_g, sd_g),
            't-loc-scale': stats.t.ppf([al / 2, 1 - al / 2], df_t, mu_t, sc_t),
            'conformal': np.quantile(ev, [al / 2, 1 - al / 2]),
        }.items():
            r = evaluate(yt, pt + ql, pt + qh, al, rng)
            print(f"{name:<12}{lev:>6.2f}{r['PICP']:>8.2f}{r['ACE']:>8.2f}{r['PINAW']:>8.4f}{r['Winkler']:>10.2f}")


if __name__ == '__main__':
    main()
