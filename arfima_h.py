"""
h-step-ahead walk-forward forecasts from the repository's SimpleARFIMA model.

The model is FITTED ON THE TRAINING PORTION ONLY (d via GPH, AR(p) via OLS) and then run
causally: the forecast issued at origin t0 for time t0+h uses only y[:t0+1].
Multi-step forecasts are produced recursively (forecasted values are fed back in).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arfima import SimpleARFIMA  # noqa: E402  (original implementation, unchanged)


def fit_arfima(series, train_end, ar_order=5, k_max=80):
    """train_end: index (exclusive) of the last series value allowed in fitting."""
    return SimpleARFIMA(ar_order=ar_order, k_max=k_max).fit(np.asarray(series[:train_end], float))


def forecast_h(model, series, origins, h):
    """Returns an array f with f[j] = forecast of series[origins[j] + h] made at origins[j]."""
    y = np.asarray(series, float)
    w = model.pi_weights               # (1-B)^d weights, w[0] = 1
    K, p, c = model.k_max, model.p, model.coef
    out = np.empty(len(origins))
    for j, t0 in enumerate(origins):
        lo = max(0, t0 - K - p - 1)
        buf = list(y[lo:t0 + 1])       # history up to and including t0
        for _ in range(h):
            arr = np.asarray(buf)
            m = len(arr)
            if m < K + p + 1:          # not enough history -> persistence fallback
                buf.append(arr[-1]); continue
            # differenced values at the last p positions
            diffs = [np.dot(w, arr[m - 1 - k - K:m - k][::-1]) for k in range(p)]
            w_hat = c[0] + np.dot(c[1:], diffs)
            y_hat = w_hat - np.dot(w[1:K + 1], arr[m - K:m][::-1])
            buf.append(y_hat)
        out[j] = buf[-1]
    return out
