"""
Shared data pipeline for the SPARC-Net revision experiments.

Fixes relative to the original real_data_pipeline.py / real_features.py / harness.py:

  1. WINDOW ALIGNMENT. The original local window was xL[t0-24 : t0] (hours t0-24 .. t0-1)
     while the target was y[t0+1]. The neural models therefore never saw the most recent
     local observation y[t0] (they solved a 2-step problem), whereas the neighbour features
     (taken at t0) and the ARFIMA stage (which conditions on y[t0]) did. Here every model
     sees exactly the same information set: local window hours t0-23 .. t0, neighbours at t0,
     target y[t0+h].

  2. NO PRIVILEGED INFORMATION FOR THE LINEAR STAGE. In the synthetic harness the ARFIMA
     stage was fitted and run on the noise-free `true_local` series, which no deployed system
     (and none of the baselines) can observe. ARFIMA is now always fitted and run on the same
     observed series the networks receive.

  3. CAUSAL GAP FILLING. The original used interpolate().ffill().bfill() over the full record,
     which fills gaps using future values. Inputs are now filled causally (forward fill with a
     limit), and test metrics are computed only on hours whose target was actually observed.

  4. MULTI-HORIZON + ANY TARGET STATION. Horizon h and the target station are parameters.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

COORDS = {
    'Aotizhongxin': (40.002, 116.397), 'Changping': (40.220, 116.230),
    'Dingling': (40.292, 116.220), 'Dongsi': (39.929, 116.417),
    'Guanyuan': (39.929, 116.339), 'Gucheng': (39.914, 116.184),
    'Huairou': (40.328, 116.628), 'Nongzhanguan': (39.937, 116.461),
    'Shunyi': (40.127, 116.655), 'Tiantan': (39.886, 116.407),
    'Wanliu': (39.987, 116.287), 'Wanshouxigong': (39.878, 116.352),
}
STATIONS = list(COORDS)
COMPASS = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135,
           'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
           'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
WIN = 24
FILL_LIMIT = 6


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _csv_path(name):
    for cand in (os.path.join(REPO, f'PRSA_Data_{name}_20130301-20170228.csv'),
                 os.path.join(REPO, 'data', 'real', f'PRSA_Data_{name}_20130301-20170228.csv')):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f'CSV for station {name} not found')


def load_station(name):
    """Returns the raw columns plus causally filled versions and an 'observed' mask."""
    df = pd.read_csv(_csv_path(name))
    df['wd_deg'] = df['wd'].map(COMPASS)
    out = pd.DataFrame({'hour': df['hour']})
    out['pm_observed'] = df['PM2.5'].notna().values
    for src, dst in (('PM2.5', 'pm'), ('WSPM', 'ws'), ('wd_deg', 'wd')):
        s = df[src].ffill(limit=FILL_LIMIT)
        # leading gaps (start of record only) cannot be filled causally; fill with first value
        s = s.fillna(s.dropna().iloc[0]) if s.isna().iloc[0] else s
        s = s.ffill()  # long gaps beyond limit: carry last value (rare; flagged via pm_observed)
        out[dst] = s.values
    return out


def build_dataset(local='Dongsi', n_neighbors=5, horizon=1):
    lat0, lon0 = COORDS[local]
    dists = {s: haversine_km(lat0, lon0, *COORDS[s]) for s in STATIONS if s != local}
    neighbors = sorted(dists, key=dists.get)[:n_neighbors]
    bearings = {s: bearing_deg(lat0, lon0, *COORDS[s]) for s in neighbors}

    L = load_station(local)
    n = len(L)
    hour = L['hour'].values
    pm = L['pm'].values.astype(float)
    ws = L['ws'].values.astype(float)
    wd = np.radians(L['wd'].values.astype(float))
    xL = np.stack([pm, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), ws], axis=1)

    xN = np.zeros((n_neighbors, n, 5))
    for i, s in enumerate(neighbors):
        nd = load_station(s)
        xN[i, :, 0] = nd['pm'].values
        xN[i, :, 1] = 1.0  # no calibration metadata in regulatory data
        xN[i, :, 2] = np.cos(wd - np.radians(bearings[s]))  # wd = direction wind blows FROM
        xN[i, :, 3] = dists[s]
        xN[i, :, 4] = ws

    t0s = np.arange(WIN - 1, n - horizon)            # window = hours t0-23 .. t0 (inclusive)
    X_local = np.stack([xL[t - WIN + 1:t + 1] for t in t0s]).astype(np.float32)
    X_neigh = np.transpose(xN[:, t0s, :], (1, 0, 2)).astype(np.float32)
    y = pm[t0s + horizon].astype(np.float32)
    y_obs = L['pm_observed'].values[t0s + horizon]

    return dict(X_local=X_local, X_neigh=X_neigh, y=y, y_observed=y_obs, t0=t0s,
                series=pm, hour=hour, horizon=horizon, local=local, neighbors=neighbors,
                distances=[dists[s] for s in neighbors])


def split(n, train=0.70, val=0.15):
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)


def normalise(d, tr):
    Xl, Xn = d['X_local'].copy(), d['X_neigh'].copy()
    ml = Xl[tr].reshape(-1, Xl.shape[-1]).mean(0); sl = Xl[tr].reshape(-1, Xl.shape[-1]).std(0) + 1e-8
    mn = Xn[tr].reshape(-1, Xn.shape[-1]).mean(0); sn = Xn[tr].reshape(-1, Xn.shape[-1]).std(0) + 1e-8
    conf = Xn[:, :, 1].copy()
    Xl = (Xl - ml) / sl
    Xn = (Xn - mn) / sn
    Xn[:, :, 1] = conf          # calibration-confidence channel stays in [0,1]
    return Xl.astype(np.float32), Xn.astype(np.float32)


def metrics(y, p):
    e = y - p
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    r2 = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(mae=mae, rmse=rmse, r2=r2)
