import numpy as np

WIN = 24

def build_windows(d):
    n = len(d['t'])
    hour = d['hour']
    sin_h = np.sin(2*np.pi*hour/24)
    cos_h = np.cos(2*np.pi*hour/24)
    ws = d['wind_speed']
    wd = d['wind_dir']

    # local pathway features xL(t) = [PM2.5L(t), sin, cos, WS(t)]  -> per-timestep, 4 channels
    xL_full = np.stack([d['obs_local'], sin_h, cos_h, ws], axis=1)  # (n,4)

    # neighbor features per neighbor i at time t: [PM2.5i, ci, cos(WD-xi), Li, WS]
    align = np.stack([np.cos(wd - d['bearing'][i]) for i in range(5)], axis=0)  # (5,n)
    xN = np.zeros((5, n, 5))
    for i in range(5):
        xN[i, :, 0] = d['obs_neighbors'][i]
        xN[i, :, 1] = d['conf_neighbors'][i]
        xN[i, :, 2] = align[i]
        xN[i, :, 3] = d['dist'][i]
        xN[i, :, 4] = ws

    target = d['true_local']  # continuous PM2.5 regression target (t+1 forecast)

    X_local, X_neigh, X_static, y, idx = [], [], [], [], []
    for t0 in range(WIN, n-1):
        X_local.append(xL_full[t0-WIN:t0])                 # (24,4)
        X_neigh.append(xN[:, t0, :])                        # (5,5)  current-time neighbor summary
        y.append(target[t0+1])
        idx.append(t0)
    return (np.array(X_local, dtype=np.float32),
            np.array(X_neigh, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(idx))

def chrono_split(n, train_frac=0.70, val_frac=0.15):
    n_train = int(n*train_frac)
    n_val = int(n*val_frac)
    return n_train, n_train+n_val
