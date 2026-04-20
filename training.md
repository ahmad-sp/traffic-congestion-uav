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

# Training with new Site with Calibration

## How data collection works:

### Step 1 — Run server normally at the new site (video or demo mode)   
```bash 
  python -m backend.main
  Every minute, metrics are automatically saved to data/traffic.db.   
```
### Step 2 — After 2+ weeks, export to CSV
```bash
  # All data
  python -m scripts.export_training_data

  # Only last 14 days for a specific junction

  python -m scripts.export_training_data --junction JCT02 --days 14   
  --out data/jct02_collected.csv
```
### Step 3 — Calibrate thresholds
  ```bash
  python -m scripts.calibrate_site --data data/collected_site_data.csv
```
### Step 4 — Restart server
  ```bash
  python -m backend.main
  ```

# Counting Line Adjust
Open config.py and modify Line 79:

python
## Change this value from 0.70 to whatever you need:
```bash
COUNTING_LINE_Y_FRACTION_LINE = float(os.getenv("COUNTING_LINE_Y", "0.50"))
```








# Training with Calibration
 ## Complete Training Pipeline

  ### Step 1: Label each extracted CSV

  Open each data/*_extracted.csv in Excel/Sheets. Add a label column. Watch the video and label every   
  row:

  ┌─────────────┬──────────────────────────────────────────────────────────┐
  │    Label    │                           When                           │
  ├─────────────┼──────────────────────────────────────────────────────────┤
  │ NORMAL      │ Free-flowing, typical traffic                            │
  ├─────────────┼──────────────────────────────────────────────────────────┤
  │ PEAK_EXCESS │ Peak-hour congestion — heavy flow, queues, slow movement │
  └─────────────┴──────────────────────────────────────────────────────────┘

  ### Step 2: Merge labeled CSVs

  python -m scripts.merge_extracted

  Combines all 17 labeled CSVs → data/real_combined.csv

  ### Step 3: Train the Autoencoder on real data
  ```bash
  python -m scripts.train_autoencoder --data data/real_combined.csv --epochs 80
  ```
  On gpu:

  ```bash
  python -m scripts.train_autoencoder --data data/real_combined.csv --epochs 80 --device cuda  
  ```

  - Trains only on your NORMAL rows (~500+ rows)
  - 80 epochs is good for small real datasets — gives the AE enough passes to learn your site's normal  
  patterns without overfitting
  - Saves: autoencoder.pt, ae_threshold.json, ae_norm_stats.json

  ### Step 4: Generate synthetic data
```bash
   python -m scripts.generate_synthetic_data --days 30
```
  Produces data/synthetic/all_arms_combined.csv with NORMAL + OFF_PEAK_JAM + PEAK_EXCESS (~86,400 rows  
  for 2 arms × 30 days).

  ### Step 5: Merge real + synthetic for LSTM
```bash
  python -m scripts.merge_lstm_training
```
  Combines your real labeled data + synthetic data → data/lstm_training.csv

  ### Step 6: Train the LSTM on combined data

```bash
  python -m scripts.train_lstm --data data/lstm_training.csv --epochs 60
```
  - 60 epochs — the combined dataset is large enough (real + synthetic) that more epochs helps the model
   see both real and synthetic patterns well
  - Both heads train: congestion detection + 10-min extreme risk forecast
  - extreme_congestion_future auto-computed from labels
  - Saves: lstm_congestion.pt, lstm_norm_stats.json

  ### Step 7: Calibrate to your real site

```bash
  python -m scripts.calibrate_site --data data/real_combined.csv
```
  - Overwrites lstm_norm_stats.json with your real feature distributions
  - Recalibrates ae_threshold.json to your site
  - Creates hourly_baseline.json for Warrant 3

  ### Step 8: Verify

  ls models_saved/

  ┌──────────────────────┬───────────┬───────────────────────────────────────────────┐
  │         File         │ From Step │                    Purpose                    │
  ├──────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ autoencoder.pt       │ 3         │ Anomaly detector (trained on real)            │
  ├──────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ ae_norm_stats.json   │ 3         │ AE normalization (real)                       │
  ├──────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ ae_threshold.json    │ 7         │ Anomaly threshold (recalibrated to real)      │
  ├──────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ lstm_congestion.pt   │ 6         │ Congestion + early warning (real + synthetic) │
  ├──────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ lstm_norm_stats.json │ 7         │ LSTM normalization (recalibrated to real)     │
  ├──────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ hourly_baseline.json │ 7         │ VPM baselines for Warrant 3                   │
  └──────────────────────┴───────────┴───────────────────────────────────────────────┘

  ---                                                                                                     Epoch guidance
                                                                                                        
  ┌─────────────┬─────────────────┬────────────────┬───────────────────────────────────────────────┐    
  │    Model    │  Dataset size   │  Recommended   │                      Why                      │      │             │                 │     epochs     │                                               │
  ├─────────────┼─────────────────┼────────────────┼───────────────────────────────────────────────┤    
  │ Autoencoder │ ~500 real       │ 80             │ Small dataset needs more passes, AE is simple │    
  │             │ NORMAL rows     │                │  (10→8→10) so low overfit risk                │      ├─────────────┼─────────────────┼────────────────┼───────────────────────────────────────────────┤
  │ LSTM        │ ~87,000         │ 60             │ Large dataset, dual-head model needs enough   │    
  │             │ combined rows   │                │ epochs to converge on both heads              │      └─────────────┴─────────────────┴────────────────┴───────────────────────────────────────────────┘
                                                                                                          If you have a GPU, add --device cuda to both training commands to speed it up. 