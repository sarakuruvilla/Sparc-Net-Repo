import numpy as np

def generate_testbed(seed=42, n_steps=4200, coupling_mult=1.0):
    rng = np.random.default_rng(seed)
    n_nodes = 6  # 1 local + 5 neighbors
    t = np.arange(n_steps)
    hour = t % 24

    # --- neighbor geometry (fixed given seed) ---
    dist = rng.uniform(0.5, 4.5, size=5)          # km, within 5x5 km grid
    bearing = rng.uniform(0, 2*np.pi, size=5)      # neighbor->local bearing xi

    # --- wind: diurnal + 9-day synoptic cycle ---
    wind_speed = (2.5 + 0.8*np.sin(2*np.pi*hour/24 - np.pi/2)
                  + 0.6*np.sin(2*np.pi*t/(9*24))
                  + rng.normal(0, 0.4, n_steps))
    wind_speed = np.clip(wind_speed, 0.2, None)
    wind_dir = np.mod(np.cumsum(rng.normal(0, 0.12, n_steps)) + 2*np.pi*t/(9*24), 2*np.pi)

    def diurnal_double_peak(h):
        # morning + evening traffic peaks (smooth, following a 24h cycle)
        m = np.exp(-0.5*((h-8)/3.2)**2)
        e = np.exp(-0.5*((h-19)/3.8)**2)
        return 15 + 20*m + 24*e

    def ar1_process(n, phi, sd, rng_local):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi*x[i-1] + rng_local.normal(0, sd)
        return x

    phi = 0.85
    base_local = diurnal_double_peak(hour)
    noise_local = ar1_process(n_steps, phi, 0.6, rng)
    true_local_self = np.clip(base_local + noise_local, 2, None)

    neighbor_true = np.zeros((5, n_steps))
    for i in range(5):
        base_i = diurnal_double_peak(np.mod(hour + rng.integers(-2, 3), 24))
        noise_i = ar1_process(n_steps, phi, 0.6, rng)
        neighbor_true[i] = np.clip(base_i + noise_i, 2, None)

    # --- wind-advected contribution from neighbors into local true signal ---
    coupling = 0.35 * coupling_mult
    advected = np.zeros(n_steps)
    mean_ws = np.maximum(wind_speed.mean(), 0.5)
    lag_hours = np.clip((dist / mean_ws).round().astype(int), 0, 6)
    for i in range(5):
        align = np.clip(np.cos(wind_dir - bearing[i]), 0, None)  # only downwind contributes
        atten = np.exp(-dist[i] / 5.0)
        for tau in range(n_steps):
            src_t = tau - lag_hours[i]
            if src_t >= 0:
                advected[tau] += coupling * align[tau] * atten * (neighbor_true[i, src_t] - neighbor_true[i].mean())

    true_local = np.clip(true_local_self + advected, 2, None)

    # --- calibration process (per node, including local) ---
    def calibration_series(n, rng_local, cal_interval=180):
        slope = np.zeros(n)
        cur = rng_local.normal(1.01, 0.01)
        drift_target = cur
        for i in range(n):
            if i % cal_interval == 0:
                cur = rng_local.normal(1.01, 0.01)
                drift_pct = rng_local.uniform(0.05, 0.08) * rng_local.choice([-1, 1])
                drift_target = cur * (1 + drift_pct)
            frac = (i % cal_interval) / cal_interval
            slope[i] = cur + frac * (drift_target - cur)
        return slope

    slope_local = calibration_series(n_steps, rng)
    obs_local = true_local * slope_local + rng.normal(0, 0.9, n_steps)

    lam, sigma_ref = 1.0, 0.05
    conf_local = np.exp(-lam * np.abs(slope_local - 1) / sigma_ref)

    obs_neighbors = np.zeros((5, n_steps))
    conf_neighbors = np.zeros((5, n_steps))
    for i in range(5):
        slope_i = calibration_series(n_steps, rng)
        obs_neighbors[i] = neighbor_true[i] * slope_i + rng.normal(0, 0.9, n_steps)
        conf_neighbors[i] = np.exp(-lam * np.abs(slope_i - 1) / sigma_ref)

    return dict(
        t=t, hour=hour, true_local=true_local, obs_local=obs_local, conf_local=conf_local,
        obs_neighbors=obs_neighbors, conf_neighbors=conf_neighbors,
        wind_speed=wind_speed, wind_dir=wind_dir, dist=dist, bearing=bearing
    )
