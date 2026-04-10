# AI Traffic Congestion Detection & Early Warning System

A real-time, AI-driven system that ingests traffic camera feeds, detects and classifies congestion events using YOLO + ByteTrack + dual ML models, evaluates rule-based warrants, and dispatches drone trigger payloads with GPS coordinates on RED alerts.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [ML Models](#5-ml-models)
6. [Warrant Engine & Alert State Machine](#6-warrant-engine--alert-state-machine)
7. [Drone Integration](#7-drone-integration)
8. [Database Schema](#8-database-schema)
9. [REST API Reference](#9-rest-api-reference)
10. [WebSocket Interface](#10-websocket-interface)
11. [Frontend](#11-frontend)
12. [Scripts Reference](#12-scripts-reference)
13. [Configuration Reference](#13-configuration-reference)
14. [Junction & Arm Configuration](#14-junction--arm-configuration)
15. [Installation — Local Development](#15-installation--local-development)
16. [Training the ML Models](#16-training-the-ml-models)
17. [Running with Real Video Sources](#17-running-with-real-video-sources)
18. [Demo Mode](#18-demo-mode)
19. [Docker Deployment](#19-docker-deployment)
20. [Labelling Real Data for Training](#20-labelling-real-data-for-training)
21. [Environment Variables Summary](#21-environment-variables-summary)

---

## 1. Project Overview

The system monitors road junction approach arms in real time. For each arm it:

1. Ingests video frames from an RTSP camera stream (or a local `.mp4` demo file).
2. Runs YOLOv8 object detection filtered to vehicle classes (car, motorcycle, bus, truck).
3. Associates detections across frames with ByteTrack multi-object tracking.
4. Aggregates per-minute traffic metrics: vehicles-per-minute (VPM), queue depth, stopped ratio, occupancy, bounding-box statistics, approach flow, and a far-zone growth rate.
5. Feeds metrics into two complementary ML models:
   - **Autoencoder** — trained on normal traffic only; high reconstruction error signals an off-peak anomaly (Case A: sudden jam).
   - **Dual-head LSTM** — trained on 60-minute rolling windows; outputs a congestion score and a 10-minute extreme-risk forecast.
6. Evaluates four warrants (W1, W2, W3, WX) to determine alert level: GREEN, AMBER, RED, or EARLY_RED.
7. On RED: compiles a drone trigger packet (GPS, severity, evidence paths) and POSTs it to a configurable webhook.
8. Persists all metrics, alerts, drone triggers, and operator feedback to SQLite (or any SQLAlchemy-compatible DB).
9. Streams live updates to the React dashboard via WebSocket.

### Congestion types

| Type | Description | ML signal |
|------|-------------|-----------|
| `OFF_PEAK_JAM` | Sudden standstill outside peak hours | Autoencoder anomaly + queue_depth > 0 |
| `PEAK_EXCESS` | Abnormally long peak-hour queue | LSTM congestion score >= 0.7 + Warrant 3 |
| `EARLY_EXTREME` | Predicted extreme congestion within 10 min | LSTM extreme risk >= 0.65, queue not yet present |

---

## 2. Repository Layout

```
traffic_system/
├── config.py                      # All configurable parameters (single source of truth)
├── docker-compose.yml
│
├── backend/
│   ├── main.py                    # FastAPI app, lifespan startup, demo simulation loop
│   ├── Dockerfile
│   ├── requirements.txt
│   │
│   ├── api/
│   │   ├── routes.py              # REST endpoints (router)
│   │   ├── websocket.py           # WebSocket manager + broadcast helpers
│   │   └── _state.py              # Shared in-process state (alert manager, warrant engines)
│   │
│   ├── pipeline/
│   │   ├── ingestion.py           # CameraReader + IngestionManager (RTSP / file)
│   │   ├── detection.py           # VehicleDetector (YOLOv8)
│   │   ├── tracking.py            # VehicleTracker (ByteTrack)
│   │   ├── metrics.py             # MetricsAggregator (per-minute aggregation)
│   │   └── counting_line.py       # Counting-line crossing logic
│   │
│   ├── models/
│   │   ├── lstm_model.py          # LSTMCongestionForecaster (dual-head)
│   │   ├── autoencoder.py         # TrafficAutoencoder + AnomalyDetector
│   │   └── inference.py           # InferenceRunner (unified LSTM + AE entry point)
│   │
│   ├── warrants/
│   │   ├── engine.py              # WarrantEngine (W1, W2, W3, WX)
│   │   └── baseline.py            # Hourly 85th-percentile VPM baseline (load/build/cache)
│   │
│   ├── alerts/
│   │   ├── manager.py             # AlertManager (state machine, debounce, EARLY_RED)
│   │   └── drone_trigger.py       # DroneTriggerManager (packet compilation + webhook POST)
│   │
│   └── db/
│       ├── models.py              # SQLAlchemy ORM models + init_db()
│       └── crud.py                # Database read/write helpers
│
├── frontend/
│   └── src/
│       ├── App.jsx                # Root layout, WebSocket wiring, arm selection
│       ├── hooks/
│       │   └── useWebSocket.js    # WebSocket hook (metrics, alerts, drone triggers)
│       └── components/
│           ├── AlertLog.jsx       # Alert table with confirm/dismiss feedback
│           ├── DroneLog.jsx       # Drone trigger log with JSON expand + evidence link
│           ├── JunctionGrid.jsx   # Collapsible junction/arm grid with badge colours
│           ├── MetricsStrip.jsx   # Live gauges: VPM, queue, LSTM score, AE score, extreme risk
│           ├── RoadMap.jsx        # React Leaflet map with arm polylines coloured by alert level
│           └── TimeSeriesChart.jsx# Recharts: VPM, baseline, congestion score, extreme risk
│
├── scripts/
│   ├── process_video_interactive.py  # Interactive ROI selection -> YOLO/ByteTrack -> CSV
│   ├── generate_synthetic_data.py    # 30-day synthetic per-minute data generator
│   ├── train_autoencoder.py          # Train autoencoder on NORMAL-only data
│   └── train_lstm.py                 # Train dual-head LSTM; save weights + norm stats
│
├── data/
│   ├── synthetic/                 # Output of generate_synthetic_data.py
│   └── hourly_baseline.json       # Built automatically from CSV on first run
│
├── models_saved/                  # Trained model weights + threshold/norm JSON
├── evidence/                      # Video clips and snapshots for RED alerts
└── logs/
```

---

## 3. System Architecture

### Component diagram

```
Camera (RTSP / file)
        |
        v
  CameraReader              <- per-camera thread, frame queue, FPS throttle
  (backend/pipeline/ingestion.py)
        |  FramePacket
        v
  VehicleDetector (YOLOv8)  <- detect vehicles, filter to ROI / counting zone
  (backend/pipeline/detection.py)
        |  Detection list
        v
  VehicleTracker (ByteTrack) <- assign persistent track IDs across frames
  (backend/pipeline/tracking.py)
        |  Track list
        v
  MetricsAggregator         <- per-frame -> per-minute rollup
  (backend/pipeline/metrics.py)
        |  MinuteMetrics dataclass
        v
  InferenceRunner           <- LSTM (seq=60 min) + Autoencoder (single timestep)
  (backend/models/inference.py)
        |  {lstm_score, extreme_congestion_risk, anomaly_score, is_anomaly}
        v
  WarrantEngine             <- W1 / W2 / W3 / WX rules -> alert_level, congestion_type
  (backend/warrants/engine.py)
        |  WarrantEngineOutput
        v
  AlertManager              <- debounce, escalation, de-escalation, EARLY_RED
  (backend/alerts/manager.py)
        |  Alert (GREEN / AMBER / RED / EARLY_RED)
        |---> DB save (crud.save_alert)
        |---> WebSocket broadcast (ws_manager.send_alert)
        `---> [if RED] DroneTriggerManager.compile_trigger -> webhook POST
```

### Startup sequence (`backend/main.py`)

1. `init_db()` — create all tables if absent.
2. `load_baseline()` — load `data/hourly_baseline.json`; if missing, build from `data/synthetic/all_arms_combined.csv` automatically.
3. Instantiate one `WarrantEngine` per arm (keyed `junction_id_arm_id`).
4. Instantiate `InferenceRunner` (loads LSTM + AE weights from `models_saved/`).
5. Wire `on_alert` / `on_early_red` callbacks on `AlertManager`.
6. If no RTSP or `DEMO_VIDEO_PATH` is set, start **demo simulation thread** replaying synthetic CSV data.
7. Otherwise, start `IngestionManager.start_all()` to open real camera readers.

---

## 4. Data Pipeline

### 4.1 Frame ingestion (`backend/pipeline/ingestion.py`)

Each arm runs a `CameraReader` in a daemon thread:

- Opens the video source with `cv2.VideoCapture` (RTSP URL or local file path).
- Calculates `frame_skip = source_fps / target_fps` to maintain `FRAME_RATE` (default 5 FPS).
- Pushes `FramePacket` objects into a bounded `queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)`.
- If the queue is full, the oldest frame is dropped (tail-drop policy).
- On RTSP disconnect, waits 2 s then reconnects automatically.
- On local file end-of-file (demo mode), seeks back to frame 0 and loops.

### 4.2 Detection (`backend/pipeline/detection.py`)

- Model: `yolov8n.pt` (auto-downloaded by Ultralytics on first run).
- Confidence threshold: `YOLO_CONFIDENCE_THRESHOLD` (default 0.45).
- Kept COCO classes: car (2), motorcycle (3), bus (5), truck (7).
- Device: `YOLO_DEVICE` (default `"cpu"`; set to `"cuda:0"` for GPU).

### 4.3 Tracking (`backend/pipeline/tracking.py`)

ByteTrack multi-object tracker:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TRACK_HIGH_THRESH` | 0.5 | Confidence for first association pass |
| `TRACK_LOW_THRESH` | 0.1 | Confidence for second association pass |
| `TRACK_MATCH_THRESH` | 0.8 | IoU threshold for track/detection matching |
| `TRACK_BUFFER` | 30 | Frames to keep a lost track (6 s at 5 FPS) |
| `TRACK_FRAME_RATE` | = `FRAME_RATE` | Tracker's internal frame rate |

### 4.4 Metrics aggregation (`backend/pipeline/metrics.py`)

`MetricsAggregator` accumulates per-frame data and produces one `MinuteMetrics` record per minute:

| Column | Description |
|--------|-------------|
| `VPM` | Vehicles crossing the counting line in the past minute |
| `queue_depth` | Near-zone stopped-vehicle count sustained for >= `QUEUE_DEPTH_SUSTAIN_SECONDS` |
| `stopped_ratio` | Fraction of tracked vehicles classified as stopped |
| `occupancy_pct` | % of frame area covered by vehicle bounding boxes |
| `mean_bbox_area` | Mean bounding-box pixel area of all tracked vehicles |
| `max_bbox_area` | Maximum bounding-box pixel area |
| `approach_flow` | Rate of vehicles entering the near zone from the far zone |
| `mean_bbox_growth_rate` | Mean per-frame bbox area delta in far zone (px^2/frame); positive = normal approach, negative = jam |
| `time_sin`, `time_cos` | Cyclical encoding of minute-of-day |
| `is_peak_hour` | 1 if current hour is in `PEAK_PERIODS`, else 0 |
| `hour_of_week` | 0 (Mon 00:00) to 167 (Sun 23:00) |

#### Zone and line definitions (all in `config.py`)

```
Frame height = H

FAR zone:    centroid_y < FAR_ZONE_Y_FRACTION * H   (far from camera, top of frame)
NEAR zone:   centroid_y > NEAR_ZONE_Y_FRACTION * H  (close to camera, bottom of frame)
Counting line: y = COUNTING_LINE_Y_FRACTION * H
```

A vehicle is **stopped** when its centroid displacement is below `STOP_THRESHOLD` px/frame for `STOP_CONSECUTIVE_FRAMES` consecutive frames.

A vehicle is **approaching** when its bbox area delta exceeds `APPROACH_THRESHOLD` px^2/frame.

#### Counting line crossing

A vehicle is counted when its track centroid crosses `COUNTING_LINE_Y_FRACTION * frame_height` (downward crossing = entering junction arm).
Source: `backend/pipeline/counting_line.py`

---

## 5. ML Models

### 5.1 Autoencoder — off-peak anomaly detector

**Architecture** (`backend/models/autoencoder.py`)

```
Input (10 features)
  -> Linear(10 -> 32) + ReLU
  -> Linear(32 -> 16) + ReLU
  -> Linear(16 -> 8)            <- bottleneck
  -> Linear(8 -> 16) + ReLU
  -> Linear(16 -> 32) + ReLU
  -> Linear(32 -> 10)           <- reconstruction
```

**Input features** (10, no `mean_bbox_growth_rate`):
`VPM, queue_depth, stopped_ratio, occupancy_pct, mean_bbox_area, max_bbox_area, approach_flow, time_sin, time_cos, is_peak_hour`

**Training** (`scripts/train_autoencoder.py`):
- Data: `data/synthetic/normal_only_combined.csv` (NORMAL rows only).
- Features are z-score normalised; stats saved to `models_saved/ae_norm_stats.json`.
- Loss: MSE reconstruction. Optimizer: Adam, lr = `AE_TRAIN_LR`.
- Best validation checkpoint saved to `models_saved/autoencoder.pt`.
- Anomaly threshold = 95th percentile of training-set reconstruction errors, saved to `models_saved/ae_threshold.json`.

**Inference** (`backend/models/inference.py`):
- Runs on every new minute's single feature vector.
- Returns `(anomaly_score: float, is_anomaly: bool)`.
- `is_anomaly = True` when `reconstruction_error > threshold`.

**What it detects**: Off-peak sudden jams (Case A). The model has never seen abnormal patterns during training, so a jam produces unusually high reconstruction error.

### 5.2 LSTM — congestion forecaster (dual-head)

**Architecture** (`backend/models/lstm_model.py`)

```
Input: (batch, seq_len=60, features=11)
  -> 2-layer LSTM(hidden=128, dropout=0.2)
  -> last hidden state (batch, 128)
       |---> Head 1 (congestion):
       |       Linear(128 -> 64) + ReLU -> Linear(64 -> 1) + Sigmoid
       |       -> congestion_score in [0, 1]
       `---> Head 2 (extreme risk):
               Linear(128 -> 32) + ReLU -> Linear(32 -> 1) + Sigmoid
               -> extreme_congestion_risk in [0, 1]
```

**Input features** (11, including `mean_bbox_growth_rate`):
`VPM, queue_depth, stopped_ratio, occupancy_pct, mean_bbox_area, max_bbox_area, approach_flow, time_sin, time_cos, is_peak_hour, mean_bbox_growth_rate`

**Training** (`scripts/train_lstm.py`):
- Data: `data/synthetic/all_arms_combined.csv` (both NORMAL and abnormal rows).
- Sliding-window sequences of length 60 (one per minute).
- Label 1 (congestion): `1.0` if row label != `"NORMAL"`.
- Label 2 (extreme): `extreme_congestion_future` — 1 if `queue_depth > 0` appears anywhere in the next 10 minutes.
- Features z-score normalised per arm; norm stats saved to `models_saved/lstm_norm_stats.json`.
- Loss: `0.6 * BCE(congestion) + 0.4 * BCE(extreme)`.
- Best validation checkpoint saved to `models_saved/lstm_congestion.pt`.

**Inference** (`backend/models/inference.py`):
- Per-camera `InferenceRunner` maintains a rolling buffer of the last 60 feature vectors.
- LSTM runs only when the buffer is full (>= 60 timesteps); `lstm_ready` is `False` until then.
- Returns `{lstm_score, extreme_congestion_risk, lstm_ready}`.

**What it detects**:
- `congestion_score >= LSTM_CONGESTION_THRESHOLD` (default 0.7) -> current or imminent Peak Excess (Case B).
- `extreme_congestion_risk >= EXTREME_CONGESTION_FORECAST_THRESHOLD` (default 0.65) with `queue_depth == 0` -> EARLY_RED (10-minute lookahead warning).

### 5.3 Saved model files

| File | Contents |
|------|----------|
| `models_saved/autoencoder.pt` | Autoencoder state dict |
| `models_saved/ae_threshold.json` | `anomaly_threshold`, percentile, mean/std/max error |
| `models_saved/ae_norm_stats.json` | Per-feature means and stds for AE normalisation |
| `models_saved/lstm_congestion.pt` | LSTM state dict |
| `models_saved/lstm_norm_stats.json` | Per-camera per-feature means and stds for LSTM normalisation |

---

## 6. Warrant Engine & Alert State Machine

### 6.1 Warrant definitions (`backend/warrants/engine.py`)

One `WarrantEngine` instance runs per arm. It is called once per minute after inference.

#### WARRANT_1 — 8-hour sustained volume

**Condition**: The hourly-average VPM exceeds `W1_VOLUME_THRESHOLD` (default 12) in all 8 of the last 8 hours.

**Fires**: AMBER (if WX is not also firing).

#### WARRANT_2 — 4-hour consecutive elevated volume

**Condition**: The hourly-average VPM exceeds `W2_VOLUME_THRESHOLD` (default 15) for 4 **consecutive** hours.

**Fires**: AMBER.

#### WARRANT_3 — Peak-hour excess

**Condition**: `current_vpm > baseline_85th[hour_of_week] x W3_PEAK_MULTIPLIER` (default 1.40).

The `baseline_85th` value is the 85th-percentile VPM for that arm at that hour-of-week, computed from historical (NORMAL) data and loaded from `data/hourly_baseline.json`.

**Fires**: AMBER, and is also a gate for WARRANT_X condition A.

#### WARRANT_X — Abnormal congestion (hard RED trigger)

Two independent conditions:

| Condition | Logic | Congestion type |
|-----------|-------|-----------------|
| A (Peak Excess) | `lstm_ready AND lstm_score >= 0.7 AND WARRANT_3 fired` | `PEAK_EXCESS` |
| B (Off-Peak Jam) | `is_anomaly AND queue_depth > 0` | `OFF_PEAK_JAM` |

Either condition fires WARRANT_X, which immediately escalates the alert to **RED** regardless of current level.

### 6.2 Alert state machine (`backend/alerts/manager.py`)

```
States:  GREEN  ->  AMBER  ->  EARLY_RED  ->  RED
               <-------------------------------

Escalation:   immediate (no debounce)
De-escalation: requires ALERT_DEESCALATE_MINUTES (default 2) consecutive GREEN minutes
Same-level re-issue: suppressed if last alert < ALERT_DEBOUNCE_MINUTES (default 5) ago
```

Level order used for comparison: `GREEN(0) < AMBER(1) < EARLY_RED(2) < RED(3)`.

**EARLY_RED** is a special level that:
- Fires when `extreme_congestion_risk >= 0.65` AND `queue_depth == 0` AND current level < RED.
- Does **not** fire warrants, does **not** trigger the drone.
- Logged to the `early_extreme_events` DB table.
- Displayed in the dashboard with a purple badge.

### 6.3 Alert lifecycle

```
WarrantEngineOutput
        |
        v
AlertManager.process_warrant_output()
  |-- Escalating?   -> issue alert immediately
  |-- Same level?   -> debounce 5 min
  `-- De-escalating? -> require 2 clean minutes first
        |
        v  Alert issued
  |-- crud.save_alert()          -> alerts table
  |-- ws_manager.send_alert()    -> WebSocket clients
  `-- if RED:
        DroneTriggerManager.compile_trigger()
          |-- crud.save_drone_trigger()    -> drone_triggers table
          `-- ws_manager.send_drone_trigger()

AlertManager.process_extreme_risk()  (called after process_warrant_output if not already RED)
  `-- if extreme_risk >= 0.65 AND queue == 0 AND level < RED:
        issue EARLY_RED alert
          |-- on_alert() callback
          `-- on_early_red() -> crud.save_early_extreme_event()
```

---

## 7. Drone Integration

### Trigger conditions

A drone trigger packet is compiled and dispatched by `DroneTriggerManager.compile_trigger()` whenever an alert reaches level **RED** (`backend/alerts/drone_trigger.py`).

### Trigger packet schema

```json
{
  "trigger_id":            "uuid4",
  "timestamp_iso":         "2026-04-09T14:32:00.000Z",
  "junction_id":           "JCT01",
  "arm_id":                "ARM_NORTH",
  "gps_lat":               51.5074,
  "gps_lon":               -0.1278,
  "congestion_type":       "OFF_PEAK_JAM",
  "severity_score":        0.872,
  "lstm_score":            0.834,
  "anomaly_score":         0.0091,
  "current_VPM":           3,
  "queue_depth":           11,
  "active_warrants":       ["WARRANT_X"],
  "evidence_clip_path":    "",
  "evidence_snapshot_path": ""
}
```

`severity_score = max(lstm_score, min(1.0, anomaly_score * 100))`

GPS coordinates are taken from `config.JUNCTIONS[junction_id]["arms"][arm_id]["gps_lat/gps_lon"]`.

### Webhook dispatch

If `DRONE_WEBHOOK_URL` is set (non-empty), the trigger packet is HTTP POSTed as JSON using `httpx` with a 10-second timeout. If the URL is empty, the trigger is only logged and stored in the database (mock/log-only mode).

### Evidence files

Evidence clips (`evidence/`) are mounted as static files at `/evidence/` on the FastAPI server. The `evidence_clip_path` and `evidence_snapshot_path` fields in the trigger packet are relative paths within this directory.

The `EVIDENCE_CLIP_SECONDS` parameter (default 30) controls how many seconds of video are preserved before a RED alert.

---

## 8. Database Schema

Source: `backend/db/models.py`

Default database: SQLite at `data/traffic.db`. Set `DATABASE_URL` env var for PostgreSQL or other.

### `minute_metrics`

One row per camera per minute.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | DATETIME | Minute timestamp (indexed) |
| `junction_id` | STRING(32) | e.g. `JCT01` |
| `arm_id` | STRING(32) | e.g. `ARM_NORTH` |
| `camera_id` | STRING(64) | `{junction_id}_{arm_id}` (indexed) |
| `VPM` | INTEGER | Vehicles per minute |
| `queue_depth` | INTEGER | Near-zone stopped vehicle count |
| `stopped_ratio` | FLOAT | Fraction of vehicles stopped |
| `occupancy_pct` | FLOAT | % frame area covered by vehicles |
| `mean_bbox_area` | FLOAT | Mean bbox pixel area |
| `max_bbox_area` | FLOAT | Max bbox pixel area |
| `approach_flow` | FLOAT | Far-to-near zone approach rate |
| `time_sin` | FLOAT | Cyclical time encoding (sin) |
| `time_cos` | FLOAT | Cyclical time encoding (cos) |
| `is_peak_hour` | INTEGER | 0 or 1 |
| `hour_of_week` | INTEGER | 0-167 |
| `mean_speed_proxy` | FLOAT | Mean centroid displacement (px/frame) |
| `extreme_congestion_risk` | FLOAT | LSTM extreme risk output (nullable) |

Composite index on `(camera_id, timestamp)`.

### `alerts`

Every alert that passes debounce and state-machine checks.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `alert_id` | STRING(36) | UUID (unique, indexed) |
| `timestamp` | DATETIME | Alert issue time (indexed) |
| `junction_id` | STRING(32) | |
| `arm_id` | STRING(32) | |
| `level` | STRING(8) | `GREEN`, `AMBER`, `EARLY_RED`, `RED` |
| `congestion_type` | STRING(32) | `OFF_PEAK_JAM`, `PEAK_EXCESS`, `EARLY_EXTREME`, or NULL |
| `active_warrants` | JSON | List of firing warrant names |
| `lstm_score` | FLOAT | |
| `anomaly_score` | FLOAT | |
| `current_vpm` | INTEGER | |
| `queue_depth` | INTEGER | |
| `confirmed` | BOOLEAN | Operator feedback (nullable = not yet reviewed) |
| `notes` | STRING(512) | Operator notes |
| `snapshot_path` | STRING(256) | Path to evidence snapshot (if any) |

### `drone_triggers`

One row per RED alert drone dispatch.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `trigger_id` | STRING(36) | UUID (unique, indexed) |
| `timestamp_iso` | STRING(64) | ISO 8601 string |
| `junction_id` | STRING(32) | |
| `arm_id` | STRING(32) | |
| `gps_lat` | FLOAT | From arm config |
| `gps_lon` | FLOAT | From arm config |
| `congestion_type` | STRING(32) | `OFF_PEAK_JAM` or `PEAK_EXCESS` |
| `severity_score` | FLOAT | Combined severity |
| `lstm_score` | FLOAT | |
| `anomaly_score` | FLOAT | |
| `current_VPM` | INTEGER | |
| `queue_depth` | INTEGER | |
| `active_warrants` | JSON | |
| `evidence_clip_path` | STRING(256) | |
| `evidence_snapshot_path` | STRING(256) | |

### `hourly_baseline`

85th-percentile VPM per arm per hour-of-week.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | |
| `junction_id` | STRING(32) | |
| `arm_id` | STRING(32) | |
| `hour_of_week` | INTEGER | 0-167 |
| `vpm_85th` | FLOAT | 85th-percentile VPM |

Unique index on `(junction_id, arm_id, hour_of_week)`.

### `operator_feedback`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | |
| `alert_id` | STRING(36) | FK to `alerts.alert_id` (indexed) |
| `timestamp` | DATETIME | |
| `confirmed` | BOOLEAN | `True` = confirmed real event, `False` = dismissed |
| `notes` | STRING(512) | |

### `early_extreme_events`

Logged each time EARLY_RED fires.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | |
| `event_id` | STRING(36) | UUID (unique, indexed) |
| `timestamp` | DATETIME | (indexed) |
| `junction_id` | STRING(32) | |
| `arm_id` | STRING(32) | |
| `extreme_congestion_risk` | FLOAT | LSTM extreme risk score |
| `mean_bbox_growth_rate` | FLOAT | Far-zone bbox growth rate at event time |
| `VPM` | INTEGER | |
| `stopped_ratio` | FLOAT | |
| `occupancy_pct` | FLOAT | |

---

## 9. REST API Reference

Base URL: `http://localhost:8000` (default).

Source: `backend/api/routes.py`

### `GET /health`

Health check.

```json
{"status": "ok", "service": "traffic-congestion-detection"}
```

---

### `GET /junctions`

List all configured junctions with arms and their current alert levels.

**Response** (array):
```json
[
  {
    "junction_id": "JCT01",
    "name": "Main Street / Oak Avenue",
    "type": "+",
    "arms": [
      {
        "arm_id": "ARM_NORTH",
        "name": "Main St Northbound Approach",
        "gps_lat": 51.5074,
        "gps_lon": -0.1278,
        "alert_level": "GREEN"
      }
    ]
  }
]
```

---

### `GET /junction/{junction_id}/status`

Per-arm live metrics and alert level for one junction.

**Path param**: `junction_id` — e.g. `JCT01`

**Response**:
```json
{
  "junction_id": "JCT01",
  "arms": {
    "ARM_NORTH": {
      "alert_level": "AMBER",
      "metrics": { "VPM": 18, "queue_depth": 3 }
    }
  }
}
```

---

### `GET /metrics/{junction_id}/{arm_id}?minutes=60`

Historical per-minute metrics. `minutes` defaults to 60, max 1440.

**Response** (array of metric rows):
```json
[
  {
    "timestamp": "2026-04-09T14:00:00",
    "VPM": 12,
    "queue_depth": 0,
    "stopped_ratio": 0.05,
    "occupancy_pct": 12.3,
    "mean_bbox_area": 3200.0,
    "max_bbox_area": 6800.0,
    "approach_flow": 9.4,
    "is_peak_hour": 1,
    "mean_speed_proxy": 4.2,
    "extreme_congestion_risk": 0.12
  }
]
```

---

### `GET /alerts?limit=50&level=RED`

Recent alerts. `limit` default 50, max 500. `level` optional filter.

**Response** (array):
```json
[
  {
    "alert_id": "uuid",
    "timestamp": "2026-04-09T14:32:00",
    "junction_id": "JCT01",
    "arm_id": "ARM_NORTH",
    "level": "RED",
    "congestion_type": "OFF_PEAK_JAM",
    "active_warrants": ["WARRANT_X"],
    "lstm_score": 0.43,
    "anomaly_score": 0.0091,
    "current_vpm": 3,
    "queue_depth": 11,
    "confirmed": null,
    "notes": ""
  }
]
```

---

### `GET /alerts/{alert_id}`

Single alert by UUID. Adds `snapshot_path` field.

---

### `GET /warrants/active`

All currently firing warrants across all arms (from in-process state).

---

### `GET /drone/triggers?limit=20`

Recent drone trigger packets. `limit` default 20, max 100.

**Response** (array):
```json
[
  {
    "trigger_id": "uuid",
    "timestamp_iso": "2026-04-09T14:32:00.000Z",
    "junction_id": "JCT01",
    "arm_id": "ARM_NORTH",
    "gps_lat": 51.5074,
    "gps_lon": -0.1278,
    "congestion_type": "OFF_PEAK_JAM",
    "severity_score": 0.872,
    "lstm_score": 0.834,
    "anomaly_score": 0.0091,
    "current_VPM": 3,
    "queue_depth": 11,
    "active_warrants": ["WARRANT_X"],
    "evidence_clip_path": "",
    "evidence_snapshot_path": ""
  }
]
```

---

### `GET /baseline/{junction_id}/{arm_id}`

168-hour baseline (hour_of_week -> vpm_85th) for one arm.

**Response**:
```json
{
  "junction_id": "JCT01",
  "arm_id": "ARM_NORTH",
  "baseline": {"0": 5.2, "1": 4.8, "167": 3.1}
}
```

---

### `POST /config`

Update a runtime threshold without restart.

**Allowed keys**: `W1_VOLUME_THRESHOLD`, `W2_VOLUME_THRESHOLD`, `W3_PEAK_MULTIPLIER`, `YOLO_CONFIDENCE_THRESHOLD`, `STOP_THRESHOLD`, `FRAME_RATE`, `ALERT_DEBOUNCE_MINUTES`, `ALERT_DEESCALATE_MINUTES`.

**Request body**:
```json
{"key": "W3_PEAK_MULTIPLIER", "value": 1.5}
```

**Response**:
```json
{"key": "W3_PEAK_MULTIPLIER", "value": 1.5}
```

---

### `GET /early-events?limit=50&junction_id=JCT01`

Recent EARLY_RED (extreme risk prediction) events.

**Response** (array):
```json
[
  {
    "event_id": "uuid",
    "timestamp": "2026-04-09T07:45:00",
    "junction_id": "JCT01",
    "arm_id": "ARM_NORTH",
    "extreme_congestion_risk": 0.71,
    "mean_bbox_growth_rate": 8.2,
    "VPM": 14,
    "stopped_ratio": 0.08,
    "occupancy_pct": 18.5
  }
]
```

---

### `POST /feedback/{alert_id}`

Submit operator feedback on an alert.

**Request body**:
```json
{"confirmed": true, "notes": "Verified via CCTV replay"}
```

**Response**:
```json
{"status": "ok", "alert_id": "uuid"}
```

---

## 10. WebSocket Interface

**Endpoint**: `ws://localhost:8000/ws/live`

Source: `backend/api/websocket.py`, `backend/main.py`

The server pushes JSON messages to all connected clients. The client may send `"ping"` to receive `"pong"`.

### Message types

#### `metrics` — per-minute metric update for one arm

```json
{
  "type": "metrics",
  "junction_id": "JCT01",
  "arm_id": "ARM_NORTH",
  "data": {
    "VPM": 18,
    "queue_depth": 3,
    "stopped_ratio": 0.12,
    "occupancy_pct": 22.1,
    "mean_bbox_area": 4200.0,
    "max_bbox_area": 8900.0,
    "approach_flow": 13.2,
    "time_sin": 0.866,
    "time_cos": -0.5,
    "is_peak_hour": 1,
    "mean_bbox_growth_rate": 42.1,
    "lstm_score": 0.34,
    "anomaly_score": 0.0018,
    "extreme_congestion_risk": 0.09,
    "alert_level": "GREEN"
  }
}
```

Broadcast every `WS_BROADCAST_INTERVAL_SECONDS` (default 5 s).

#### `alert` — new alert issued

```json
{
  "type": "alert",
  "junction_id": "JCT01",
  "arm_id": "ARM_NORTH",
  "data": {
    "alert_id": "uuid",
    "level": "RED",
    "congestion_type": "OFF_PEAK_JAM",
    "active_warrants": ["WARRANT_X"],
    "timestamp": "2026-04-09T14:32:00Z"
  }
}
```

#### `drone_trigger` — drone dispatch on RED alert

```json
{
  "type": "drone_trigger",
  "junction_id": "JCT01",
  "arm_id": "ARM_NORTH",
  "data": { "...full DroneTriggerPacket fields..." }
}
```

### Frontend hook

`frontend/src/hooks/useWebSocket.js` manages reconnection and distributes incoming messages into `metrics`, `alerts`, and `droneTriggers` state slices used by `App.jsx`.

---

## 11. Frontend

Built with React + Vite + Tailwind CSS.

Source: `frontend/src/`

API base URL configured via `VITE_API_URL` environment variable (defaults to `http://localhost:8000`).

### Components

#### `App.jsx`

Root component. Manages:
- WebSocket connection via `useWebSocket` hook.
- Junction list (fetched from `GET /junctions` on mount).
- Alert list (fetched from `GET /alerts?limit=50` on mount, merged with WebSocket alerts).
- Selected arm state — drives metrics strip and time-series chart.
- `metricsHistory` — fetched from `GET /metrics/{junction_id}/{arm_id}?minutes=60` when arm selection changes.

Layout (top to bottom):
1. Header bar with live/disconnected WebSocket indicator.
2. `RoadMap` — full-width Leaflet map.
3. `JunctionGrid` — collapsible junction/arm panels.
4. `MetricsStrip` + `TimeSeriesChart` (shown only when an arm is selected, side-by-side on large screens).
5. `AlertLog` + `DroneLog` (side-by-side on large screens).

---

#### `RoadMap.jsx`

React Leaflet map (OpenStreetMap tiles).

- Junction centre markers (blue `CircleMarker`).
- Arm polylines from junction centre to arm GPS position, coloured by alert level.
- Arm endpoint `CircleMarker` with popup showing arm name, status, VPM, and queue depth.
- Click on any polyline or arm marker calls `onSelectArm`.

Alert level colours:

| Level | Colour |
|-------|--------|
| `GREEN` | `#22c55e` |
| `AMBER` | `#f59e0b` |
| `EARLY_RED` | `#8B5CF6` |
| `RED` | `#ef4444` |

---

#### `JunctionGrid.jsx`

Collapsible grid of junction cards (1 column on mobile, 2 on md+).

- Each junction card is bordered and tinted by its highest-severity arm level.
- Junction-level badge = highest level across all arms (RED > EARLY_RED > AMBER > GREEN).
- Each arm row shows its level badge, name, and live VPM / queue depth.
- Click arm row calls `onSelectArm`.
- Click junction header to toggle collapse.

---

#### `MetricsStrip.jsx`

Live metrics panel for the selected arm. Updates on every WebSocket `metrics` message.

Displayed widgets:
- **Vehicles Per Minute** — blue gauge bar (max 50).
- **Queue Depth** — amber gauge bar (max 20 vehicles).
- **Congestion Score (LSTM)** — score bar with threshold line at 0.7.
- **Anomaly Score (AE)** — score bar with threshold line at 0.004.
- **Extreme Risk (10 min)** — purple-coloured gauge; threshold line at 0.65.

---

#### `TimeSeriesChart.jsx`

Recharts `ComposedChart` of the last 60 minutes for the selected arm.

Series plotted:

| Series | Axis | Colour | Type |
|--------|------|--------|------|
| VPM | Left | Blue | Line |
| 85th pct baseline | Left | Gray dashed | Line |
| Congestion Score | Right (0-1) | Red area | Area |
| Extreme Risk | Right (0-1) | Purple dashed | Line |

Alert events appear as vertical `ReferenceLine` markers coloured by their level.

---

#### `AlertLog.jsx`

Scrollable alert table (max height 320 px).

Columns: Time, Junction, Arm, Level, Type, Warrants, Status.

- Unconfirmed RED alerts pulse and show **Confirm** / **Dismiss** buttons.
- Clicking a row expands a detail panel with LSTM score, anomaly score, VPM, and queue depth.
- Confirm/Dismiss buttons call `POST /feedback/{alert_id}`.

---

#### `DroneLog.jsx`

Scrollable drone trigger log.

Each entry shows:
- Trigger ID prefix, junction/arm, congestion type, severity score, VPM, timestamp.
- **Details** button expands a raw JSON view of the full trigger packet.
- If `evidence_clip_path` is set, a link to `/evidence/{path}` is shown.

---

## 12. Scripts Reference

All scripts are in `traffic_system/scripts/`. Run them from the `traffic_system/` directory.

### `process_video_interactive.py` — extract metrics from real video

Processes a handycam `.mp4` file through YOLO + ByteTrack and exports per-minute metrics to CSV for training.

```bash
python scripts/process_video_interactive.py --video path/to/video.mp4 [--output-dir data/]
```

**Workflow**:

1. Opens the first frame in a `cv2` window labelled **"ROI Selection"**.
2. You draw a polygon ROI by clicking.
3. The video is processed frame-by-frame; only vehicles whose bottom edge intersects the polygon are tracked.
4. Metrics are aggregated per minute and written to `data/<video_stem>_extracted.csv`.

**ROI controls**:

| Key | Action |
|-----|--------|
| Left-click | Add polygon vertex |
| ENTER | Confirm polygon (requires >= 3 points) |
| BACKSPACE | Undo last point |
| `r` | Reset all points |
| ESC | Cancel and exit |

The ROI is per-session only — it is not saved to disk. Zone boundaries (`NEAR_ZONE_Y_FRACTION`, `FAR_ZONE_Y_FRACTION`) and counting line position (`COUNTING_LINE_Y_FRACTION`) are set in `config.py` and apply globally.

**Output CSV columns**: timestamp (frame-derived), junction_id, arm_id, camera_id, VPM, queue_depth, stopped_ratio, occupancy_pct, mean_bbox_area, max_bbox_area, approach_flow, mean_bbox_growth_rate, time_sin, time_cos, is_peak_hour, hour_of_week.

---

### `generate_synthetic_data.py` — generate training data

Generates 30 days of realistic per-minute traffic data for every configured arm.

```bash
# With anomaly injection (default — for LSTM training)
python -m scripts.generate_synthetic_data [--days 30] [--seed 42]

# Normal-only (for autoencoder training)
python -m scripts.generate_synthetic_data --no-anomalies
```

**Output files** in `data/synthetic/`:

| File | Contents |
|------|----------|
| `JCT01_ARM_NORTH.csv` | Per-arm data (one file per arm) |
| `all_arms_combined.csv` | All arms concatenated (used by LSTM training) |
| `normal_only_combined.csv` | Normal-only data (used by autoencoder training) |

**Traffic patterns** generated:

| Period | VPM range | Hours |
|--------|-----------|-------|
| Morning peak | 18-25 + noise(sigma=3) | 07:30-09:00 |
| Evening peak | 20-28 + noise(sigma=4) | 17:00-19:00 |
| Daytime | 8-14 | 09:00-17:00 |
| Night | 1-5 | 22:00-06:00 |
| Weekend scaling | x0.6 | Sat/Sun all day |

**Injected abnormal events** (when `--inject-anomalies` / default):

| Type | VPM | Queue depth | Duration | Timing |
|------|-----|-------------|----------|--------|
| `OFF_PEAK_JAM` | 0-2 | 8-15 | 20-40 min | 1-3 random off-peak windows |
| `PEAK_EXCESS` | 35-45 | 10-20 | 60-120 min | 1-3 random peak windows |

A 10-minute pre-jam `mean_bbox_growth_rate` ramp is added before each jam onset (linear slowdown from ~50 to ~5 px^2/frame), providing the signal for the LSTM extreme-risk head.

**`extreme_congestion_future` column**: automatically computed — value is 1 for any minute where `queue_depth > 0` appears anywhere in the next 10 minutes.

---

### `train_autoencoder.py` — train the anomaly detector

```bash
python -m scripts.train_autoencoder \
    [--data data/synthetic/normal_only_combined.csv] \
    [--epochs 50] \
    [--device cpu]
```

- Filters to `label == "NORMAL"` rows (safety re-filter).
- 80/20 train/val split. Saves best validation checkpoint.
- Computes anomaly threshold at `AE_ANOMALY_PERCENTILE` (95th) percentile of training errors.
- Saves: `models_saved/autoencoder.pt`, `models_saved/ae_threshold.json`, `models_saved/ae_norm_stats.json`.

---

### `train_lstm.py` — train the congestion forecaster

```bash
python -m scripts.train_lstm --data data/synthetic/all_arms_combined.csv --epochs 50 --device cpu

```

- Groups by `camera_id`, creates 60-step sliding-window sequences per arm.
- 80/20 train/val split across all sequences (shuffled).
- Dual-head BCE loss: `0.6 x congestion + 0.4 x extreme`.
- Saves: `models_saved/lstm_congestion.pt`, `models_saved/lstm_norm_stats.json`.

---

## 13. Configuration Reference

All parameters are in `traffic_system/config.py`. No hardcoded thresholds appear anywhere else.

Parameters marked **(env)** can be overridden with an environment variable at runtime.

### Paths

| Key | Default | Description |
|-----|---------|-------------|
| `PROJECT_ROOT` | `traffic_system/` | Resolved parent of `config.py` |
| `DATA_DIR` | `traffic_system/data/` | CSV data, baseline JSON, SQLite DB |
| `MODEL_DIR` | `traffic_system/models_saved/` | Trained model weights and stats |
| `EVIDENCE_DIR` | `traffic_system/evidence/` | Video clips and snapshots |
| `SYNTHETIC_DATA_DIR` | `traffic_system/data/synthetic/` | Synthetic CSV output |
| `LOG_DIR` | `traffic_system/logs/` | Log files |

### Database

| Key | Default | Description |
|-----|---------|-------------|
| `DATABASE_URL` **(env)** | `sqlite:///data/traffic.db` | SQLAlchemy connection string |

### Frame ingestion

| Key | Default | Description |
|-----|---------|-------------|
| `FRAME_RATE` **(env: FRAME_RATE)** | `5` | Target FPS to process per camera |
| `DEMO_VIDEO_PATH` **(env)** | `""` | Fallback local `.mp4` when no RTSP is configured |
| `FRAME_QUEUE_MAXSIZE` | `128` | Per-camera frame buffer capacity |

### YOLO detection

| Key | Default | Description |
|-----|---------|-------------|
| `YOLO_MODEL_NAME` | `"yolov8n.pt"` | Ultralytics model name (auto-downloaded) |
| `YOLO_CONFIDENCE_THRESHOLD` **(env: YOLO_CONF)** | `0.45` | Minimum detection confidence |
| `YOLO_TARGET_CLASSES` | `[2, 3, 5, 7]` | COCO IDs: car, motorcycle, bus, truck |
| `YOLO_DEVICE` **(env)** | `"cpu"` | PyTorch device string (`"cpu"`, `"cuda:0"`, etc.) |

### Tracking (ByteTrack)

| Key | Default | Description |
|-----|---------|-------------|
| `TRACK_HIGH_THRESH` | `0.5` | Confidence for first association pass |
| `TRACK_LOW_THRESH` | `0.1` | Confidence for second association pass |
| `TRACK_MATCH_THRESH` | `0.8` | IoU threshold for track matching |
| `TRACK_BUFFER` | `30` | Frames to keep a lost track |
| `TRACK_FRAME_RATE` | `= FRAME_RATE` | Tracker frame rate |

### Motion & queue analysis

| Key | Default | Description |
|-----|---------|-------------|
| `STOP_THRESHOLD` **(env)** | `2.0` | px/frame below which a vehicle is stopped |
| `STOP_CONSECUTIVE_FRAMES` | `3` | Frames stopped before classified as stopped |
| `APPROACH_THRESHOLD` **(env)** | `5.0` | Bbox area delta (px^2/frame) indicating approach |
| `NEAR_ZONE_Y_FRACTION` | `0.6` | Centroid y > this fraction of frame_height = near zone |
| `FAR_ZONE_Y_FRACTION` | `0.4` | Centroid y < this fraction = far zone |
| `FAR_ZONE_UPPER_Y_RATIO` | `0.4` | Upper boundary for far-zone bbox growth rate calculation |
| `COUNTING_LINE_Y_FRACTION` **(env: COUNTING_LINE_Y)** | `0.5` | Counting line position (fraction of frame height) |
| `QUEUE_DEPTH_SUSTAIN_SECONDS` | `30` | Seconds a vehicle must be stopped to count as queued |

### LSTM model

| Key | Default | Description |
|-----|---------|-------------|
| `LSTM_SEQUENCE_LENGTH` | `60` | Input window (timesteps / minutes) |
| `LSTM_NUM_FEATURES` | `11` | Features per timestep |
| `LSTM_HIDDEN_SIZE` | `128` | LSTM hidden state size |
| `LSTM_NUM_LAYERS` | `2` | Stacked LSTM layers |
| `LSTM_DROPOUT` | `0.2` | Dropout between layers |
| `LSTM_FC_HIDDEN` | `64` | FC head hidden size (congestion head) |
| `LSTM_CONGESTION_THRESHOLD` | `0.7` | Score >= this = congested |
| `EXTREME_CONGESTION_FORECAST_THRESHOLD` **(env)** | `0.65` | Extreme risk score >= this triggers EARLY_RED |
| `EXTREME_CONGESTION_FORECAST_WINDOW` | `10` | Lookahead minutes for extreme label |
| `LSTM_LOSS_WEIGHT_CONGESTION` | `0.6` | Loss weight for congestion head |
| `LSTM_LOSS_WEIGHT_EXTREME` | `0.4` | Loss weight for extreme risk head |
| `LSTM_TRAIN_LR` | `1e-3` | Training learning rate |
| `LSTM_TRAIN_BATCH_SIZE` | `32` | Training batch size |
| `LSTM_TRAIN_EPOCHS` | `50` | Training epochs |
| `LSTM_INFERENCE_INTERVAL_SECONDS` | `60` | Inference cadence (1 per minute) |

### Autoencoder model

| Key | Default | Description |
|-----|---------|-------------|
| `AE_INPUT_DIM` | `10` | Number of input features |
| `AE_ENCODER_DIMS` | `[32, 16, 8]` | Encoder layer sizes (bottleneck = 8) |
| `AE_DECODER_DIMS` | `[16, 32, 10]` | Decoder layer sizes |
| `AE_TRAIN_LR` | `1e-3` | Training learning rate |
| `AE_TRAIN_BATCH_SIZE` | `32` | Training batch size |
| `AE_TRAIN_EPOCHS` | `50` | Training epochs |
| `AE_ANOMALY_PERCENTILE` | `95` | Percentile of training errors for threshold |
| `AE_INFERENCE_INTERVAL_SECONDS` | `60` | Inference cadence |

### Warrant engine

| Key | Default | Description |
|-----|---------|-------------|
| `W1_VOLUME_THRESHOLD` **(env)** | `12` | VPM threshold for Warrant 1 (8-hour sustained) |
| `W2_VOLUME_THRESHOLD` **(env)** | `15` | VPM threshold for Warrant 2 (4-hour consecutive) |
| `W3_PEAK_MULTIPLIER` **(env)** | `1.40` | Current VPM must exceed baseline x this for W3 |
| `W1_HOURS` | `8` | Lookback hours for Warrant 1 |
| `W2_HOURS` | `4` | Lookback hours for Warrant 2 |

### Alert system

| Key | Default | Description |
|-----|---------|-------------|
| `ALERT_DEBOUNCE_MINUTES` | `5` | Min minutes before same-level re-issue |
| `ALERT_DEESCALATE_MINUTES` | `2` | Consecutive clean minutes before de-escalation |

### Drone trigger

| Key | Default | Description |
|-----|---------|-------------|
| `DRONE_WEBHOOK_URL` **(env)** | `""` | POST target; empty = mock/log only |
| `EVIDENCE_CLIP_SECONDS` | `30` | Seconds of video to retain before RED alert |

### Peak periods

| Key | Default | Description |
|-----|---------|-------------|
| `PEAK_PERIODS` | `[(7,9), (17,19)]` | Morning and evening peak hour ranges (24h, half-open) |

Helper: `config.is_peak_hour(hour: int) -> bool`

### Synthetic data generator

| Key | Default | Description |
|-----|---------|-------------|
| `SYNTH_DAYS` | `30` | Days to generate |
| `SYNTH_DEFAULT_ARMS` | `2` | Default number of arms |
| `SYNTH_MORNING_PEAK_VPM` | `(18, 25)` | VPM range, morning peak |
| `SYNTH_MORNING_PEAK_NOISE_STD` | `3` | Gaussian noise std, morning peak |
| `SYNTH_MORNING_PEAK_HOURS` | `(7.5, 9.0)` | Hour range for morning peak |
| `SYNTH_EVENING_PEAK_VPM` | `(20, 28)` | VPM range, evening peak |
| `SYNTH_EVENING_PEAK_NOISE_STD` | `4` | Gaussian noise std, evening peak |
| `SYNTH_EVENING_PEAK_HOURS` | `(17.0, 19.0)` | Hour range for evening peak |
| `SYNTH_DAYTIME_VPM` | `(8, 14)` | VPM range, daytime |
| `SYNTH_DAYTIME_HOURS` | `(9.0, 17.0)` | Hour range for daytime |
| `SYNTH_NIGHT_VPM` | `(1, 5)` | VPM range, night |
| `SYNTH_NIGHT_HOURS_START` | `22.0` | Night starts (fractional hour) |
| `SYNTH_NIGHT_HOURS_END` | `6.0` | Night ends (fractional hour) |
| `SYNTH_WEEKEND_SCALE` | `0.6` | Weekend VPM multiplier |
| `SYNTH_OFFPEAK_JAM_VPM` | `(0, 2)` | VPM during off-peak jam |
| `SYNTH_OFFPEAK_JAM_DURATION_MIN` | `(20, 40)` | Jam duration range (minutes) |
| `SYNTH_OFFPEAK_JAM_QUEUE_DEPTH` | `(8, 15)` | Queue depth during off-peak jam |
| `SYNTH_OFFPEAK_JAM_STOPPED_RATIO` | `0.85` | Stopped vehicle ratio during off-peak jam |
| `SYNTH_PEAK_EXCESS_VPM` | `(35, 45)` | VPM during peak excess event |
| `SYNTH_PEAK_EXCESS_QUEUE_DEPTH` | `(10, 20)` | Queue depth during peak excess |
| `SYNTH_PEAK_EXCESS_DURATION_MIN` | `(60, 120)` | Duration range (minutes) |

### FastAPI / server

| Key | Default | Description |
|-----|---------|-------------|
| `API_HOST` **(env)** | `"0.0.0.0"` | Uvicorn bind host |
| `API_PORT` **(env)** | `8000` | Uvicorn port |
| `CORS_ORIGINS` **(env)** | `"http://localhost:3000,http://localhost:5173"` | Comma-separated allowed origins |
| `WS_BROADCAST_INTERVAL_SECONDS` | `5` | WebSocket push cadence |

### Frontend

| Key | Default | Description |
|-----|---------|-------------|
| `FRONTEND_PORT` **(env)** | `3000` | Frontend dev server port |
| `API_BASE_URL` **(env)** | `http://localhost:8000` | REST base URL (server-side reference) |
| `WS_BASE_URL` **(env)** | `ws://localhost:8000` | WebSocket base URL (server-side reference) |

### Logging

| Key | Default | Description |
|-----|---------|-------------|
| `LOG_LEVEL` **(env)** | `"INFO"` | Python logging level |
| `LOG_FORMAT` | `"%(asctime)s | %(name)s | %(levelname)s | %(message)s"` | Log format string |

### Junction definitions

| Key | Description |
|-----|-------------|
| `JUNCTIONS` | Dict keyed by junction ID; each entry has `name`, `type` (+/T/L), `arms` dict |
| `get_all_camera_ids()` | Returns `["{junction_id}_{arm_id}", ...]` for all configured arms |
| `get_arm_config(junction_id, arm_id)` | Returns the arm config dict (raises `KeyError` if not found) |

---

## 14. Junction & Arm Configuration

Source: `config.py`

The default configuration ships with two junctions:

| Junction ID | Name | Type | Arms |
|-------------|------|------|------|
| `JCT01` | Main Street / Oak Avenue | `+` (crossroads) | ARM_NORTH, ARM_SOUTH, ARM_EAST, ARM_WEST |
| `JCT02` | High Road / Park Lane | `T` (T-junction) | ARM_NORTH, ARM_SOUTH, ARM_EAST |

Camera ID format: `{junction_id}_{arm_id}` — e.g. `JCT01_ARM_NORTH`.

### Adding a new junction

1. Add an entry to `JUNCTIONS` in `config.py`:

```python
"JCT03": {
    "name": "Station Road / Bridge Street",
    "type": "T",
    "arms": {
        "ARM_NORTH": {
            "name": "Station Rd Northbound",
            "gps_lat": 51.520,
            "gps_lon": -0.155,
            "rtsp_url": os.getenv("JCT03_ARM_NORTH_RTSP", ""),
            "counting_line_y": COUNTING_LINE_Y_FRACTION,
        },
    },
},
```

2. Set the RTSP environment variable for each arm: `JCT03_ARM_NORTH_RTSP=rtsp://192.168.1.50/stream`.
3. Generate new synthetic training data (if using synthetic training) and retrain models.
4. Restart the backend.

### RTSP URL convention

`{JUNCTION_ID}_{ARM_ID}_RTSP` — e.g. `JCT01_ARM_NORTH_RTSP=rtsp://user:pass@192.168.1.10/stream1`.

If an arm's `rtsp_url` is empty and `DEMO_VIDEO_PATH` is set, the demo file is used for that arm. If both are empty, the arm is skipped by the real ingestion pipeline (but demo simulation will still process it from synthetic CSV data).

---

## 15. Installation — Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- pip / virtualenv

### Backend

```bash
cd traffic_system

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Optional: install OpenCV with GUI for process_video_interactive.py
# (the headless version does not support imshow windows)
pip install opencv-python
```

Dependencies installed:

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn[standard]` | ASGI server |
| `sqlalchemy` | ORM / database |
| `pydantic` | Request/response validation |
| `numpy`, `pandas` | Numerical computation and CSV handling |
| `torch` | PyTorch (LSTM + AE models) |
| `ultralytics` | YOLOv8 (auto-downloads `yolov8n.pt` on first use) |
| `opencv-python-headless` | Frame capture and image processing |
| `httpx` | HTTP client for drone webhook |
| `websockets` | WebSocket support |

### Frontend

```bash
cd traffic_system/frontend
npm install
```

### Quick start

```bash
# Terminal 1 — backend
cd traffic_system
source venv/bin/activate
python -m backend.main

# Terminal 2 — frontend
cd traffic_system/frontend
npm run dev
```

Open `http://localhost:3000` (or the port shown by Vite).

If no synthetic data exists, the backend will log a warning and demo simulation will not run. See [Section 18 — Demo Mode](#18-demo-mode) to generate data first.

---

## 16. Training the ML Models

Models must be trained before meaningful ML scores are available. Untrained models (missing weight files) are silently used as random networks — the system will still run but scores will be noise.

### Step 1 — Generate synthetic data

```bash
cd traffic_system
source venv/bin/activate

# Generate with anomaly injection — produces all_arms_combined.csv and normal_only_combined.csv
python -m scripts.generate_synthetic_data --days 30 --inject-anomalies
```

Output:
```
data/synthetic/JCT01_ARM_NORTH.csv
data/synthetic/JCT01_ARM_SOUTH.csv
data/synthetic/JCT01_ARM_EAST.csv
data/synthetic/JCT01_ARM_WEST.csv
data/synthetic/JCT02_ARM_NORTH.csv
data/synthetic/JCT02_ARM_SOUTH.csv
data/synthetic/JCT02_ARM_EAST.csv
data/synthetic/all_arms_combined.csv
data/synthetic/normal_only_combined.csv
```

### Step 2 — Train the autoencoder

```bash
python -m scripts.train_autoencoder \
    --data data/synthetic/normal_only_combined.csv \
    --epochs 50 \
    --device cpu
```

Expected output files:
```
models_saved/autoencoder.pt
models_saved/ae_threshold.json
models_saved/ae_norm_stats.json
```

### Step 3 — Train the LSTM

```bash
python -m scripts.train_lstm \
    --data data/synthetic/all_arms_combined.csv \
    --epochs 50 \
    --device cpu
```

Expected output files:
```
models_saved/lstm_congestion.pt
models_saved/lstm_norm_stats.json
```

### Step 4 — Baseline is automatic

The baseline is built automatically from `data/synthetic/all_arms_combined.csv` on the first backend startup if `data/hourly_baseline.json` is not present. No separate script is needed — just ensure the synthetic CSV exists before the first run.

Source: `backend/warrants/baseline.py:load_baseline()`

### GPU training

Add `--device cuda:0` to both training scripts. PyTorch must have CUDA support installed.

---

## 17. Running with Real Video Sources

### RTSP cameras

Set environment variables for each arm before starting the backend:

```bash
export JCT01_ARM_NORTH_RTSP="rtsp://admin:password@192.168.1.10:554/stream1"
export JCT01_ARM_SOUTH_RTSP="rtsp://admin:password@192.168.1.11:554/stream1"
```

Start the backend:

```bash
python -m backend.main
```

The `IngestionManager` will start one `CameraReader` thread per arm that has a non-empty `rtsp_url`.

### Local video file (single file, all arms)

```bash
export DEMO_VIDEO_PATH="/path/to/recording.mp4"
python -m backend.main
```

This uses the same file for all arms that lack an explicit RTSP URL.

### Extracting real training data

Use `process_video_interactive.py` to extract per-minute metrics from any handycam recording:

```bash
python scripts/process_video_interactive.py \
    --video /path/to/recording.mp4 \
    --output-dir data/
```

The output CSV (`data/<stem>_extracted.csv`) has the same column schema as synthetic data. You must manually add a `label` column with values `NORMAL`, `OFF_PEAK_JAM`, or `PEAK_EXCESS` before using it for training. See [Section 20](#20-labelling-real-data-for-training) for details.

---

## 18. Demo Mode

If no RTSP URL or `DEMO_VIDEO_PATH` is configured, the backend automatically runs a **demo simulation** thread (`_run_demo_simulation` in `backend/main.py`).

The simulation replays `data/synthetic/all_arms_combined.csv` row-by-row, feeding each minute's metrics through the full pipeline:

1. `WarrantEngine.push_vpm()` — update rolling VPM history.
2. `InferenceRunner.push_metrics()` then `.run_inference()` — get ML scores.
3. `WarrantEngine.evaluate()` — get alert level.
4. `AlertManager.process_warrant_output()` — issue alerts if warranted.
5. `AlertManager.process_extreme_risk()` — check for EARLY_RED.
6. `_state.update_latest_metrics()` — update in-memory state for REST API.
7. WebSocket broadcast every 5 rows.

Each row is processed with a 10 ms sleep (~100x speed relative to real time).

### Running the demo

```bash
# Step 1: generate data
python -m scripts.generate_synthetic_data

# Step 2: train models
python -m scripts.train_autoencoder
python -m scripts.train_lstm

# Step 3: start backend (auto-detects no video sources, runs demo)
python -m backend.main

# Step 4: start frontend
cd frontend && npm run dev
```

Navigate to `http://localhost:3000`. You should see arms colouring on the map, metrics updating in the JunctionGrid, alerts appearing in AlertLog, and drone triggers in DroneLog when RED alerts fire.

---

## 19. Docker Deployment

Source: `docker-compose.yml`, `backend/Dockerfile`

### Build and run

```bash
cd traffic_system
docker-compose up --build
```

Services:

| Service | Port | Description |
|---------|------|-------------|
| `backend` | 8000 | FastAPI + Uvicorn |
| `frontend` | 3000 | React app |

### Volumes

```yaml
backend:
  volumes:
    - ./data:/app/data              # CSV files, SQLite DB, baseline JSON
    - ./models_saved:/app/models_saved   # Trained model weights
    - ./evidence:/app/evidence      # Evidence clips / snapshots
```

Models and data must be trained/generated **on the host** before running Docker, or the backend will start with untrained models.

### Environment variables in Docker

Edit `docker-compose.yml` or use a `.env` file:

```yaml
environment:
  - DATABASE_URL=sqlite:///data/traffic.db
  - LOG_LEVEL=INFO
  - YOLO_DEVICE=cpu
  - DEMO_VIDEO_PATH=
  - DRONE_WEBHOOK_URL=https://your-drone-api.example.com/trigger
  - JCT01_ARM_NORTH_RTSP=rtsp://192.168.1.10/stream
```

### Backend Dockerfile summary

```dockerfile
FROM python:3.11-slim
# System deps: libgl1-mesa-glx (OpenCV), libglib2.0-0
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 20. Labelling Real Data for Training

There is no interactive labelling tool. Labelling is done by manually editing the extracted CSV.

### Procedure

1. Run `process_video_interactive.py` on a recorded video to extract metrics.
2. Open the output CSV in a spreadsheet editor or Python/pandas.
3. Add a `label` column with one of these values:

| Value | Meaning |
|-------|---------|
| `NORMAL` | Normal traffic — used for both LSTM and AE training |
| `OFF_PEAK_JAM` | Sudden standstill outside peak hours |
| `PEAK_EXCESS` | Abnormally long peak-hour congestion |

4. Compute the `extreme_congestion_future` column (required for LSTM training):

```python
import pandas as pd

df = pd.read_csv("data/my_video_extracted.csv")

lookahead = 10  # minutes
df["extreme_congestion_future"] = 0
for i in range(len(df) - lookahead):
    if df["queue_depth"].iloc[i + 1 : i + 1 + lookahead].max() > 0:
        df.at[i, "extreme_congestion_future"] = 1

df.to_csv("data/my_video_labelled.csv", index=False)
```

5. Use the labelled CSV for training:

```bash
# LSTM (uses all labels)
python -m scripts.train_lstm --data data/my_video_labelled.csv

# Autoencoder (filters to NORMAL internally)
python -m scripts.train_autoencoder --data data/my_video_labelled.csv
```

---

## 21. Environment Variables Summary

| Variable | Used in | Default | Description |
|----------|---------|---------|-------------|
| `DATABASE_URL` | `config.py` | `sqlite:///data/traffic.db` | SQLAlchemy DB URL |
| `FRAME_RATE` | `config.py` | `5` | FPS to process per camera |
| `DEMO_VIDEO_PATH` | `config.py` | `""` | Fallback local video file |
| `YOLO_CONF` | `config.py` | `0.45` | YOLO confidence threshold |
| `YOLO_DEVICE` | `config.py` | `"cpu"` | PyTorch device |
| `STOP_THRESHOLD` | `config.py` | `2.0` | Stopped vehicle px/frame threshold |
| `APPROACH_THRESHOLD` | `config.py` | `5.0` | Approach bbox delta threshold |
| `COUNTING_LINE_Y` | `config.py` | `0.5` | Counting line Y fraction |
| `EXTREME_CONGESTION_FORECAST_THRESHOLD` | `config.py` | `0.65` | LSTM extreme risk threshold |
| `W1_VOLUME_THRESHOLD` | `config.py` | `12` | Warrant 1 VPM threshold |
| `W2_VOLUME_THRESHOLD` | `config.py` | `15` | Warrant 2 VPM threshold |
| `W3_PEAK_MULTIPLIER` | `config.py` | `1.40` | Warrant 3 baseline multiplier |
| `DRONE_WEBHOOK_URL` | `config.py` | `""` | Drone trigger webhook POST URL |
| `API_HOST` | `config.py` | `"0.0.0.0"` | FastAPI bind host |
| `API_PORT` | `config.py` | `8000` | FastAPI port |
| `CORS_ORIGINS` | `config.py` | `"http://localhost:3000,http://localhost:5173"` | Comma-separated CORS origins |
| `FRONTEND_PORT` | `config.py` | `3000` | Frontend port (reference) |
| `API_BASE_URL` | `config.py` | `http://localhost:8000` | REST API base URL |
| `WS_BASE_URL` | `config.py` | `ws://localhost:8000` | WebSocket base URL |
| `LOG_LEVEL` | `config.py` | `"INFO"` | Python logging level |
| `JCT01_ARM_NORTH_RTSP` | `config.py` | `""` | RTSP URL for JCT01 ARM_NORTH |
| `JCT01_ARM_SOUTH_RTSP` | `config.py` | `""` | RTSP URL for JCT01 ARM_SOUTH |
| `JCT01_ARM_EAST_RTSP` | `config.py` | `""` | RTSP URL for JCT01 ARM_EAST |
| `JCT01_ARM_WEST_RTSP` | `config.py` | `""` | RTSP URL for JCT01 ARM_WEST |
| `JCT02_ARM_NORTH_RTSP` | `config.py` | `""` | RTSP URL for JCT02 ARM_NORTH |
| `JCT02_ARM_SOUTH_RTSP` | `config.py` | `""` | RTSP URL for JCT02 ARM_SOUTH |
| `JCT02_ARM_EAST_RTSP` | `config.py` | `""` | RTSP URL for JCT02 ARM_EAST |
| `VITE_API_URL` | Frontend (Vite) | `http://localhost:8000` | REST API base URL for React app |
