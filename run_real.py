"""
One experiment = (target station, horizon, model, seed). Saves per-sample test predictions,
so Diebold-Mariano tests and interval forecasts can be computed afterwards.

usage:
  python run_real.py --station Dongsi --horizon 1 --model sparc --seed 7
models:
  persistence      y(t0)                                  (no seed needed)
  seasonal         y(t0+h-24)                             (no seed needed)
  arfima           ARFIMA linear stage alone              (no seed needed)
  gru              single-node GRU (paper baseline)
  arfima_gru       ARFIMA + single-node GRU on residual   (decomposition WITHOUT spatial info)
  spatial          spatial attention, raw target          (spatial info WITHOUT decomposition)
  sparc            ARFIMA + spatial attention on residual (SPARC-Net, calibration term off)
"""
import os, sys, json, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import common, baselines, arfima_h  # noqa: E402

OUT = os.path.join(HERE, 'results_rev')
os.makedirs(OUT, exist_ok=True)
NEURAL = {'gru', 'arfima_gru', 'spatial', 'sparc'}


def train_predict(build_fn, Xtr, Xva, Xte, ytr, yva, seed, epochs=30, patience=6, batch=128):
    import tensorflow as tf
    mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-8
    m = build_fn(seed)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    m.fit(Xtr, (ytr - mu) / sd, validation_data=(Xva, (yva - mu) / sd),
          epochs=epochs, batch_size=batch, verbose=0, callbacks=[es])
    pva = m.predict(Xva, verbose=0).ravel() * sd + mu
    pte = m.predict(Xte, verbose=0).ravel() * sd + mu
    return pva, pte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--station', default='Dongsi', choices=common.STATIONS)
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--model', required=True,
                    choices=['persistence', 'seasonal', 'arfima', 'gru', 'arfima_gru', 'spatial', 'sparc'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    tag = f'{a.station}_h{a.horizon}_{a.model}' + (f'_s{a.seed}' if a.model in NEURAL else '')
    path = os.path.join(OUT, tag + '.npz')
    if os.path.exists(path):
        print('exists, skipping:', tag); return
    t_start = time.time()

    d = common.build_dataset(a.station, horizon=a.horizon)
    y, t0, s = d['y'], d['t0'], d['series']
    tr, va, te = common.split(len(y))

    # linear stage: fitted on training data only, run causally on the observed series
    need_L = a.model in {'arfima', 'arfima_gru', 'sparc'}
    if need_L:
        arf = arfima_h.fit_arfima(s, train_end=t0[tr.stop - 1] + 1)
        L = arfima_h.forecast_h(arf, s, t0, a.horizon)

    if a.model == 'persistence':
        pred_va, pred_te = baselines.persistence(s, t0[va], a.horizon), baselines.persistence(s, t0[te], a.horizon)
    elif a.model == 'seasonal':
        pred_va = baselines.seasonal_persistence(s, t0[va], a.horizon)
        pred_te = baselines.seasonal_persistence(s, t0[te], a.horizon)
    elif a.model == 'arfima':
        pred_va, pred_te = L[va], L[te]
    else:
        from models import build_single_node_gru, build_es_bigru_net  # original architectures
        Xl, Xn = common.normalise(d, tr)
        spatial_fn = lambda q: build_es_bigru_net(q, use_calibration=False)
        if a.model in {'gru', 'arfima_gru'}:
            fn, X = build_single_node_gru, (Xl[tr], Xl[va], Xl[te])
        else:
            fn, X = spatial_fn, ([Xl[tr], Xn[tr]], [Xl[va], Xn[va]], [Xl[te], Xn[te]])
        if a.model in {'arfima_gru', 'sparc'}:
            r = y - L
            pva, pte = train_predict(fn, *X, r[tr], r[va], a.seed)
            pred_va, pred_te = pva + L[va], pte + L[te]
        else:
            pred_va, pred_te = train_predict(fn, *X, y[tr], y[va], a.seed)

    mask_te = d['y_observed'][te]
    res = common.metrics(y[te][mask_te], pred_te[mask_te])
    np.savez(path, y_val=y[va], pred_val=pred_va, mask_val=d['y_observed'][va],
             y_test=y[te], pred_test=pred_te, mask_test=mask_te, t0_test=t0[te])
    res.update(station=a.station, horizon=a.horizon, model=a.model, seed=a.seed,
               n_test_scored=int(mask_te.sum()), seconds=round(time.time() - t_start, 1))
    with open(os.path.join(OUT, 'summary.jsonl'), 'a') as f:
        f.write(json.dumps(res) + '\n')
    print(json.dumps(res))


if __name__ == '__main__':
    main()
