import sys, time, json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import tensorflow as tf
from harness import prep_data
from models import build_single_node_gru, build_cnn_bigru_hard_npi, build_es_bigru_net
from arfima import SimpleARFIMA

data = prep_data()
te = data['te']
Xl,Xn,hard,y = data['Xl'],data['Xn'],data['hard'],data['y']
n_test = Xl[te].shape[0]

def bench_batched(model, inputs, reps=20):
    _ = model.predict(inputs, verbose=0, batch_size=256)  # warmup
    times=[]
    for _ in range(reps):
        t0=time.perf_counter()
        model.predict(inputs, verbose=0, batch_size=256)
        times.append((time.perf_counter()-t0)*1000)
    per_sample = np.array(times)/n_test
    return float(per_sample.mean()), float(per_sample.std())

results={}
m = build_single_node_gru(7)
results['single_gru'] = bench_batched(m, Xl[te])
m = build_cnn_bigru_hard_npi(7)
results['cnn_bigru_hard'] = bench_batched(m, [Xl[te], hard[te]])
m = build_es_bigru_net(7, use_calibration=True)
results['sparc_attention_stage'] = bench_batched(m, [Xl[te], Xn[te]])

arfima = SimpleARFIMA(ar_order=5, k_max=80).fit(data['y'][:2946])
t0=time.perf_counter()
for i in range(500):
    arfima.forecast_series(data['y'], 2946+(i%600), 2946+(i%600))
arfima_ms = (time.perf_counter()-t0)/500*1000

out = dict(nn_batched_per_sample=results, arfima_ms=arfima_ms,
           sparc_total_ms=results['sparc_attention_stage'][0]+arfima_ms,
           n_test=int(n_test))
print(json.dumps(out, indent=2))
with open('latency_results.json','w') as f:
    json.dump(out, f, indent=2)
