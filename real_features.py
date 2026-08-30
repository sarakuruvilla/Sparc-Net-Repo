import numpy as np

WIN = 24

def build_windows_real(d):
    n = d['n']
    xL_full = d['xL']; xN = d['xN']; target = d['target']
    X_local, X_neigh, y = [], [], []
    for t0 in range(WIN, n-1):
        X_local.append(xL_full[t0-WIN:t0])
        X_neigh.append(xN[:, t0, :])
        y.append(target[t0+1])
    return (np.array(X_local, dtype=np.float32), np.array(X_neigh, dtype=np.float32),
            np.array(y, dtype=np.float32))
