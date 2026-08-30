import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

COORDS = {
    'Aotizhongxin': (40.002, 116.397), 'Changping': (40.220, 116.230),
    'Dingling': (40.292, 116.220), 'Dongsi': (39.929, 116.417),
    'Guanyuan': (39.929, 116.339), 'Gucheng': (39.914, 116.184),
    'Huairou': (40.328, 116.628), 'Nongzhanguan': (39.937, 116.461),
    'Shunyi': (40.127, 116.655), 'Tiantan': (39.886, 116.407),
    'Wanliu': (39.987, 116.287), 'Wanshouxigong': (39.878, 116.352),
}
COMPASS = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
           'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2-lat1); dlmb = np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    # bearing from point1 to point2
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2-lon1)
    x = np.sin(dl)*np.cos(p2)
    y = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return (np.degrees(np.arctan2(x,y)) + 360) % 360

def load_station(name):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'real', f'PRSA_Data_{name}_20130301-20170228.csv'))
    df['wd_deg'] = df['wd'].map(COMPASS)
    df = df[['year','month','day','hour','PM2.5','WSPM','wd_deg']].copy()
    df['PM2.5'] = df['PM2.5'].interpolate(limit=6).ffill().bfill()
    df['WSPM'] = df['WSPM'].interpolate(limit=6).ffill().bfill()
    df['wd_deg'] = df['wd_deg'].interpolate(limit=6).ffill().bfill()
    return df

def build_real_dataset(local='Dongsi', n_neighbors=5, window=24):
    all_stations = list(COORDS.keys())
    lat0, lon0 = COORDS[local]
    dists = {s: haversine_km(lat0, lon0, *COORDS[s]) for s in all_stations if s != local}
    neighbors = sorted(dists, key=dists.get)[:n_neighbors]
    bearings = {s: bearing_deg(lat0, lon0, *COORDS[s]) for s in neighbors}

    local_df = load_station(local)
    n = len(local_df)
    hour = local_df['hour'].values
    sin_h = np.sin(2*np.pi*hour/24); cos_h = np.cos(2*np.pi*hour/24)
    ws_local = local_df['WSPM'].values
    wd_local = np.radians(local_df['wd_deg'].values)
    pm_local = local_df['PM2.5'].values

    xL = np.stack([pm_local, sin_h, cos_h, ws_local], axis=1)

    xN = np.zeros((n_neighbors, n, 5))
    for i, s in enumerate(neighbors):
        ndf = load_station(s)
        align = np.cos(wd_local - np.radians(bearings[s]))
        xN[i,:,0] = ndf['PM2.5'].values
        xN[i,:,1] = 1.0  # no calibration metadata available for regulatory-grade data -> confidence fixed at 1
        xN[i,:,2] = align
        xN[i,:,3] = dists[s]
        xN[i,:,4] = ws_local

    return dict(xL=xL, xN=xN, target=pm_local, n=n, local=local, neighbors=neighbors,
                distances=[dists[s] for s in neighbors])

if __name__ == '__main__':
    d = build_real_dataset()
    print('local:', d['local'])
    print('neighbors (nearest 5):', list(zip(d['neighbors'], [f'{x:.1f}km' for x in d['distances']])))
    print('n timesteps:', d['n'])
    print('xL shape', d['xL'].shape, 'xN shape', d['xN'].shape)
