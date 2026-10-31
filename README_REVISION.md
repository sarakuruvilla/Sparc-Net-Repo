# SPARC-Net — revision experiments

This folder adds the experiments needed for a resubmission. The original files in the
repository root are imported unchanged (models.py, arfima.py, data_gen.py).

## Two problems found in the original pipeline (fix these before anything else)

**1. The local window skipped the current hour.** `real_features.py` and `features.py` used
`xL[t0-24:t0]` (hours t0-24 … t0-1) to predict `y[t0+1]`. The neural networks therefore never
saw the latest local reading and were really solving a 2-hour-ahead problem, while the
neighbour features (taken at t0) and the ARFIMA stage (which conditions on y[t0]) did see the
current hour. On Dongsi (seed 7) the effect is:

| Model (Dongsi, h = 1, test RMSE, µg/m³) | Original window | Corrected window |
|---|---|---|
| Single-node GRU | 34.91 (paper, 6-seed mean) | 21.92 |
| Spatial attention alone | 26.44 (paper) | 20.91 |
| Persistence, y(t0) | — | 23.37 |
| ARFIMA alone | — | 22.25 |
| SPARC-Net (ARFIMA + spatial attention) | 21.06 (paper) | 20.65 |

With aligned inputs SPARC-Net still has the lowest error on this run, but the margin over the
single-node GRU is about 6 % (not 39.7 %) and about 1 % over spatial attention alone. These are
single-seed numbers; the full grid below is needed before any claim is written.

**2. In the synthetic testbed the ARFIMA stage used the noise-free truth.** `harness.py` fits
and runs ARFIMA on `true_local`, which no deployed sensor (and none of the baselines) can
observe. Seed 7, native coupling:

| Configuration | Single-node GRU | SPARC-Net |
|---|---|---|
| Original (ARFIMA on true_local, old window) | 1.441 | 0.836 |
| ARFIMA on observed series, old window | 1.472 | 1.456 |
| ARFIMA on observed series, corrected window | 1.318 | 1.157 |

This is why the synthetic gain survived the 0× coupling control: it came from privileged
access to the true series, not from spatial transport.

Minor: `load_station` used `interpolate().ffill().bfill()` over the full record (fills gaps
with future values), and the README's `src/ scripts/ data/real/` layout does not match the
flat repository, so the published commands fail as written.

## Files

| File | Purpose |
|---|---|
| `common.py` | Corrected data pipeline: aligned windows, causal gap filling, any station, any horizon, test scoring only on observed hours |
| `arfima_h.py` | h-step walk-forward forecasts from the original `SimpleARFIMA` (fit on training data only) |
| `baselines.py` | Persistence and same-hour-yesterday persistence |
| `run_real.py` | One run = station × horizon × model × seed; saves per-sample predictions |
| `dm_test.py` | Diebold–Mariano tests (HLN correction, Holm adjustment) + performance tables |
| `intervals.py` | Gaussian / t-location-scale / conformal intervals on the **final** forecast, sized on validation errors |
| `run_synthetic_fixed.py` | Synthetic ablation with both leaks removed (`--legacy` reproduces the original) |
| `run_all.sh` | Full grid: 12 stations × horizons 1, 6, 12, 24 × 7 models × 5 seeds |

Models compared: `persistence`, `seasonal`, `arfima`, `gru`, `arfima_gru` (decomposition
without spatial information), `spatial` (spatial information without decomposition), `sparc`.

## Running

```bash
pip install -r ../requirements.txt
cd revision
python run_real.py --station Dongsi --horizon 1 --model sparc --seed 7   # single run
STATIONS="Dongsi Tiantan" HORIZONS="1 6" SEEDS="7 17 27" bash run_all.sh  # subset
python dm_test.py                                                        # tables + DM tests
python intervals.py --station Dongsi --horizon 1 --model sparc
```

Outputs go to `revision/results_rev/` (`summary.jsonl`, per-run `.npz`, `performance_table.csv`,
`dm_tests.csv`). Neural runs take a few minutes each on CPU; the full grid is best split across
processes (one station each) or run on a GPU/Colab.
