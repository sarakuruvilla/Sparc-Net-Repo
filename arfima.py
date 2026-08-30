import numpy as np
from scipy.special import gammaln

def frac_diff_weights(d, k_max):
    # pi_k for (1-B)^d = sum_k pi_k B^k ; pi_0=1, pi_k = pi_{k-1} * (k-1-d)/k
    w = np.zeros(k_max+1)
    w[0] = 1.0
    for k in range(1, k_max+1):
        w[k] = w[k-1] * (k - 1 - d) / k
    return w

def gph_estimate_d(y, m_frac=0.5):
    """Geweke-Porter-Hudak estimator of fractional differencing parameter d."""
    n = len(y)
    y = y - y.mean()
    freq = np.fft.rfftfreq(n)[1:]
    per = (np.abs(np.fft.rfft(y))**2 / (2*np.pi*n))[1:]
    m = max(8, int(n**m_frac))
    m = min(m, len(freq)-1)
    lam = freq[:m]
    I = per[:m]
    X = np.log(4*np.sin(lam/2)**2)
    Yv = np.log(I + 1e-12)
    X1 = np.column_stack([np.ones_like(X), X])
    beta, *_ = np.linalg.lstsq(X1, Yv, rcond=None)
    d_hat = -beta[1] / 2.0
    return float(np.clip(d_hat, 0.01, 0.49))

def apply_frac_diff(y, d, k_max=100):
    w = frac_diff_weights(d, k_max)
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(k_max, n):
        out[t] = np.dot(w, y[t-k_max:t+1][::-1])
    return out, w

class SimpleARFIMA:
    """ARFIMA(p,d,q) approximated as: fractional differencing (causal, truncated)
    followed by a low-order AR model fit via least squares (Yule-Walker style OLS)
    on the differenced series. One-step-ahead forecasts are produced causally
    using only information available up to time t (true walk-forward), then
    inverted back to level space via the fractional-integration recursion."""

    def __init__(self, ar_order=5, k_max=100):
        self.p = ar_order
        self.k_max = k_max

    def fit(self, y_train):
        self.d = gph_estimate_d(y_train)
        w_diff, _ = apply_frac_diff(np.asarray(y_train), self.d, self.k_max)
        w_diff = w_diff[~np.isnan(w_diff)]
        # fit AR(p) on differenced series via OLS
        p = self.p
        Xrows, Yrows = [], []
        for i in range(p, len(w_diff)):
            Xrows.append(w_diff[i-p:i][::-1])
            Yrows.append(w_diff[i])
        X = np.array(Xrows); Yv = np.array(Yrows)
        X1 = np.column_stack([np.ones(len(X)), X])
        self.coef, *_ = np.linalg.lstsq(X1, Yv, rcond=None)
        self.pi_weights = frac_diff_weights(self.d, self.k_max)
        return self

    def forecast_series(self, y_full, start_idx, end_idx):
        """Causal one-step-ahead forecasts for y_full[start_idx+1 .. end_idx+1]
        using true history y_full[:t+1] at each step t (walk-forward, no leakage
        of future values, matching real-time deployment)."""
        n = len(y_full)
        w = self.pi_weights
        k_max = self.k_max
        p = self.p
        # precompute differenced series causally over the whole span we need
        diff_series = np.full(n, np.nan)
        for t in range(k_max, end_idx+2):
            diff_series[t] = np.dot(w, y_full[t-k_max:t+1][::-1])

        forecasts = np.full(n, np.nan)
        for t0 in range(start_idx, end_idx+1):
            # forecast differenced value at t0+1 using AR(p) on diff_series[t0-p+1 .. t0]
            hist = diff_series[t0-p+1:t0+1][::-1]
            if np.any(np.isnan(hist)):
                forecasts[t0+1] = y_full[t0]
                continue
            w_hat = self.coef[0] + np.dot(self.coef[1:], hist)
            # invert fractional differencing: y(t) = w(t) - sum_{k=1}^{K} pi_k * y(t-k)
            k_max_eff = min(k_max, t0+1)
            past_y = y_full[t0+1-k_max_eff:t0+1][::-1]
            correction = np.dot(w[1:k_max_eff+1], past_y)
            y_hat = w_hat - correction
            forecasts[t0+1] = y_hat
        return forecasts
