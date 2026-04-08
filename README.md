# AI Traffic Congestion Detection & Early Warning System

A production-ready system that uses roadside cameras, computer vision, and machine learning to detect abnormal traffic congestion and trigger drone response.

## How It Works

Each road arm approaching a junction has a camera mounted on a pole ~200m away, facing **toward oncoming traffic** (away from the junction). The system processes video feeds to detect vehicles, track their movement, and determine whether traffic is flowing normally or congesting abnormally.

### Camera Geometry

```
    Far (top of frame)          Vehicles appear small here — normal flow
         |
         |  Vehicles move downward toward camera
         |  Bounding boxes grow larger as they approach
         |
    ─────┼───── Counting Line (y = 50% of frame)
         |
         |  Near zone — large bounding boxes
         |  If vehicles are stopped here, the queue
         |  has backed up past the camera position
         |
    Near (bottom of frame)      Camera is here, looking back up the road
```

### The Warrant Framework (Non-ML Explanation)

The system uses a **warrant framework** — a structured set of rules that determine when traffic is abnormal enough to trigger an alert. Think of warrants like escalating thresholds:

| Warrant | What It Checks | Alert Level | Plain English |
|---------|---------------|-------------|---------------|
| **Warrant 1** | Is traffic volume above a threshold for 8 straight hours? | AMBER | "This road has been busy all day" |
| **Warrant 2** | Is traffic volume elevated for 4 consecutive hours? | AMBER | "This road has been unusually busy for half a shift" |
| **Warrant 3** | Is current traffic volume 40% above the historical average for this time of week? | AMBER | "Right now is much busier than this time usually is" |
| **Warrant X** | Is the AI confirming abnormal congestion? | **RED** | "Something is wrong — this isn't normal traffic" |

**Warrant X** (the hard alert) fires in two cases:

- **Case A — Off-Peak Sudden Jam**: The anomaly detector (autoencoder) flags an unusual pattern AND vehicles are queuing. Example: a crash at 2 AM causing a backup.
- **Case B — Abnormally Long Peak Queue**: The LSTM forecaster predicts high congestion AND current traffic exceeds the historical baseline by 40%+. Example: a Monday morning with 50% more traffic than usual.

When Warrant X fires, the system compiles a **drone trigger packet** with GPS coordinates, evidence clips, and congestion severity — ready for drone dispatch.

### Alert Levels

- **GREEN**: Normal traffic. No warrants active.
- **AMBER**: Heavy traffic detected (Warrant 1, 2, or 3). Monitor closely.
- **EARLY RED**: Early extreme congestion predicted within 10 minutes. The LSTM's dual-head detects far-zone bbox growth rate slowdown — vehicles upstream are decelerating before the queue physically reaches the camera. No drone trigger or warrant activation; serves as an advance warning.
- **RED**: Abnormal congestion confirmed (Warrant X). Drone trigger compiled.

## Architecture

```
Camera (RTSP) → Frame Ingestion → YOLOv8n Detection → ByteTrack Tracking
                                                            ↓
                                               Per-Minute Metrics
                                                            ↓
                                          ┌─── LSTM Forecaster (Case B)
                                          ├─── Autoencoder Detector (Case A)
                                          └─── Warrant Engine (Rules 1/2/3/X)
                                                            ↓
                                               Alert Manager (Debounce)
                                                            ↓
                                          ┌─── Dashboard (React + WebSocket)
                                          ├─── Drone Trigger Packet
                                          └─── Database (SQLite)
```

## Quick Start

### 1. Generate Synthetic Data

```bash
cd traffic_system
python -m scripts.generate_synthetic_data
```

### 2. Train Models

```bash
python -m scripts.train_lstm --epochs 30
python -m scripts.train_autoencoder --epochs 30
```

### 3. Start Backend (Demo Mode)

```bash
pip install -r backend/requirements.txt
python -m backend.main
```

The backend starts in **demo mode** when no RTSP streams are configured — it replays synthetic data through the full pipeline.

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_RATE` | 5 | FPS to process from video |
| `YOLO_CONFIDENCE_THRESHOLD` | 0.45 | Detection confidence cutoff |
| `STOP_THRESHOLD` | 2.0 px/frame | Below this = vehicle stopped |
| `LSTM_CONGESTION_THRESHOLD` | 0.7 | Score above this = congested |
| `W1_VOLUME_THRESHOLD` | 12 VPM | Warrant 1 volume threshold |
| `W2_VOLUME_THRESHOLD` | 15 VPM | Warrant 2 volume threshold |
| `W3_PEAK_MULTIPLIER` | 1.40 | Warrant 3 baseline multiplier |
| `ALERT_DEBOUNCE_MINUTES` | 5 | Min time between same alerts |

## Junction Configuration

Junctions and arms are defined in `config.py`:

```python
JUNCTIONS = {
    "JCT01": {
        "name": "Main Street / Oak Avenue",
        "type": "+",  # crossroads: L, T, or +
        "arms": {
            "ARM_NORTH": {
                "gps_lat": 51.5074,
                "gps_lon": -0.1278,
                "rtsp_url": "rtsp://...",  # or leave empty for demo
            },
            ...
        },
    },
}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/junctions` | All junctions with status |
| GET | `/junction/{id}/status` | Per-arm live metrics |
| GET | `/metrics/{jid}/{aid}?minutes=60` | Historical metrics |
| GET | `/alerts?limit=50&level=RED` | Alert log |
| GET | `/warrants/active` | Currently firing warrants |
| GET | `/drone/triggers` | Drone trigger log |
| GET | `/early-events?limit=50` | Early extreme congestion events |
| GET | `/baseline/{jid}/{aid}` | Hourly baseline |
| POST | `/config` | Update thresholds |
| POST | `/feedback/{alert_id}` | Operator feedback |
| WS | `/ws/live` | Live metrics stream |

## Using with Video

The system supports three video source modes: **demo (synthetic data)**, **local video files**, and **live RTSP streams**.

### Mode 1 — Demo (No Video)

When no video sources are configured, the backend replays synthetic CSV data through the full pipeline. This is the default — just run `python -m backend.main`.

### Mode 2 — Local Video File

Point all cameras at a single `.mp4` file using the `DEMO_VIDEO_PATH` environment variable. The file loops automatically.

```bash
# Linux / macOS
DEMO_VIDEO_PATH=/path/to/traffic_video.mp4 python -m backend.main

# Windows (PowerShell)
$env:DEMO_VIDEO_PATH="C:\path\to\traffic_video.mp4"; python -m backend.main
```

The video should be filmed from a **roadside pole facing oncoming traffic** (vehicles moving toward the camera). Any standard traffic cam footage works — the system will detect vehicles, track them, and compute metrics from the video.

### Mode 3 — Live RTSP Streams

Set RTSP URLs per arm via environment variables:

```bash
# One variable per camera, named {JUNCTION}_{ARM}_RTSP
export JCT01_ARM_NORTH_RTSP="rtsp://admin:pass@192.168.1.100:554/stream1"
export JCT01_ARM_SOUTH_RTSP="rtsp://admin:pass@192.168.1.101:554/stream1"
# ... set as many as needed

python -m backend.main
```

Or edit `config.py` directly and put the RTSP URL in the `"rtsp_url"` field for each arm:

```python
"ARM_NORTH": {
    "name": "Main St Northbound Approach",
    "gps_lat": 51.5074,
    "gps_lon": -0.1278,
    "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
},
```

The system auto-detects the source FPS and downsamples to the configured `FRAME_RATE` (default 5 FPS). Dropped RTSP connections are automatically retried.

### Camera Requirements

- Mount ~200m from the junction, on a pole, facing **away** from the junction (toward oncoming traffic)
- Vehicles should move from top of frame (far, small) toward bottom (near, large)
- Resolution: 720p+ recommended
- Frame rate: 15+ FPS source (system downsamples to 5 FPS)

## Training Models on Google Colab

The LSTM and Autoencoder can be trained on Colab with GPU acceleration, then the checkpoint files are downloaded and placed into the local `models_saved/` directory.

### Step 1 — Upload Project to Colab

Create a new Colab notebook and upload the necessary files:

```python
# Option A: Clone from GitHub (if your repo is pushed)
!git clone https://github.com/YOUR_USERNAME/uav-claude.git
%cd uav-claude/traffic_system

# Option B: Upload as zip
# 1. Zip traffic_system/ locally
# 2. Upload to Colab via the file browser (left sidebar)
# 3. Unzip:
!unzip traffic_system.zip
%cd traffic_system
```

### Step 2 — Install Dependencies

```python
!pip install torch numpy pandas
```

> You do **not** need `ultralytics`, `opencv`, `fastapi`, or other backend deps — training only uses `torch`, `numpy`, and `pandas`.

### Step 3 — Generate Synthetic Data

```python
!python -m scripts.generate_synthetic_data
```

This creates CSV files in `data/synthetic/`. Takes ~1 minute.

### Step 4 — Train LSTM (with GPU)

```python
# Make sure GPU runtime is enabled: Runtime → Change runtime type → GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")  # should print True

!python -m scripts.train_lstm --epochs 50 --device cuda
```

Training produces two files in `models_saved/`:
- `lstm_congestion.pt` — model weights (~850 KB)
- `lstm_norm_stats.json` — per-camera feature normalization stats

### Step 5 — Train Autoencoder (with GPU)

```python
!python -m scripts.train_autoencoder --epochs 50 --device cuda
```

Training produces three files in `models_saved/`:
- `autoencoder.pt` — model weights
- `ae_threshold.json` — anomaly threshold (95th percentile)
- `ae_norm_stats.json` — feature normalization stats

### Step 6 — Download Model Files

```python
from google.colab import files

files.download('models_saved/lstm_congestion.pt')
files.download('models_saved/lstm_norm_stats.json')
files.download('models_saved/autoencoder.pt')
files.download('models_saved/ae_threshold.json')
files.download('models_saved/ae_norm_stats.json')
```

### Step 7 — Place Files Locally

Copy all 5 downloaded files into your local project:

```
traffic_system/
  models_saved/
    lstm_congestion.pt          ← from Colab
    lstm_norm_stats.json        ← from Colab
    autoencoder.pt              ← from Colab
    ae_threshold.json           ← from Colab
    ae_norm_stats.json          ← from Colab
```

Then start the system normally — it will load the trained models automatically:

```bash
python -m backend.main
```

### Complete Colab Notebook (Copy-Paste)

```python
# === Cell 1: Setup ===
!git clone https://github.com/YOUR_USERNAME/uav-claude.git
%cd uav-claude/traffic_system
!pip install torch numpy pandas

# === Cell 2: Generate training data ===
!python -m scripts.generate_synthetic_data

# === Cell 3: Train LSTM ===
!python -m scripts.train_lstm --epochs 50 --device cuda

# === Cell 4: Train Autoencoder ===
!python -m scripts.train_autoencoder --epochs 50 --device cuda

# === Cell 5: Download all model files ===
from google.colab import files
for f in ['lstm_congestion.pt', 'lstm_norm_stats.json',
          'autoencoder.pt', 'ae_threshold.json', 'ae_norm_stats.json']:
    files.download(f'models_saved/{f}')
```

## Docker

```bash
docker-compose up --build
```

Backend: http://localhost:8000 | Frontend: http://localhost:3000
