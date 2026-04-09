# Training with Real Video Data

All commands are run from the `traffic_system/` directory.

---

## 1. Extract Features from Videos

Use the batch script to process a folder of raw `.mp4` recordings:

```bash
python scripts/process_video_interactive.py --folder ./videos/ --output-dir ./data/
```

For a single file:

```bash
python scripts/process_video_interactive.py --video ./videos/site_a.mp4 --output-dir ./data/
```

For each video the script will prompt for three things:

| Prompt | Purpose |
|--------|---------|
| **Junction** | Selects the junction from `config.py` (e.g. `JCT01`) |
| **Arm** | Selects the specific approach arm (e.g. `ARM_NORTH`) |
| **Recording start time** (`YYYY-MM-DD HH:MM`) | Anchors all timestamps to real wall-clock time so that `is_peak_hour`, `time_sin`, `time_cos`, and `hour_of_week` are computed correctly per minute |

Each video produces a CSV in `data/<video_name>_extracted.csv`.

---

## 2. Draw the ROI Carefully

After the prompts, a window opens showing the first frame. Click to draw a polygon.

> **WARNING: The ROI is the single most important quality control step.**
>
> Draw the polygon to cover **only the target approach lane**. If the ROI overlaps opposing traffic, a cross-street, a junction box, or parked vehicles, every vehicle in those areas will be counted. This directly inflates `VPM`, `queue_depth`, `occupancy_pct`, and `stopped_ratio` — corrupting every feature in the extracted CSV and making the resulting training data unusable.

Controls: **left-click** to add a vertex · **Backspace** to undo · **r** to reset · **Enter** to confirm (minimum 3 points).

---

## 3. Prepare the Combined Training CSV

After extraction, combine all CSVs and add a `label` column before training.

```bash
# Combine all extracted CSVs
python - <<'EOF'
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in glob.glob("data/*_extracted.csv")]
pd.concat(dfs, ignore_index=True).to_csv("data/real_combined.csv", index=False)
EOF
```

Open `data/real_combined.csv` and add a `label` column to every row. The two training scripts expect these exact values:

| Value | When to use |
|-------|-------------|
| `NORMAL` | Free-flow traffic, no incident |
| `OFF_PEAK_JAM` | Queue or jam outside peak hours |
| `PEAK_EXCESS` | Volume or queue significantly above normal peak levels |

The autoencoder trains on `NORMAL` rows only; the LSTM uses all three labels.

---

## 4. Train the Models

### Autoencoder (anomaly detector)

Trains on `NORMAL`-labelled rows only. Outputs `models_saved/autoencoder.pt`, `ae_threshold.json`, and `ae_norm_stats.json`.

```bash
python -m scripts.train_autoencoder \
    --data data/real_combined.csv \
    --epochs 50 \
    --device cpu
```

### LSTM (congestion forecaster)

Trains on all labels. Outputs `models_saved/lstm_congestion.pt` and `lstm_norm_stats.json`.

```bash
python -m scripts.train_lstm \
    --data data/real_combined.csv \
    --epochs 50 \
    --device cpu
```

Use `--device cuda:0` if a GPU is available. Use `--epochs` to adjust training length; the scripts save the best checkpoint (lowest validation loss) automatically.
