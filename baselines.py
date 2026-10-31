"""Non-learned reference forecasts that any PM2.5 model must be compared against."""
import numpy as np


def persistence(series, origins, h):
    """y_hat(t0+h) = y(t0)."""
    return np.asarray(series, float)[origins]


def seasonal_persistence(series, origins, h, period=24):
    """y_hat(t0+h) = y(t0+h-period): same hour on the previous day."""
    s = np.asarray(series, float)
    idx = origins + h - period
    out = s[np.clip(idx, 0, None)]
    return out
