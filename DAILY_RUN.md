# Daily Run — Quick Reference

## First time only

```bash
cd traffic_system
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

python -m scripts.generate_synthetic_data
python -m scripts.train_autoencoder
python -m scripts.train_lstm
```

## Every day

```bash
# Terminal 1
cd traffic_system && source venv/bin/activate
python -m backend.main

# Terminal 2
cd traffic_system/frontend && npm run dev
```

Open **http://localhost:3000**

## With real cameras

```bash
export JCT01_ARM_NORTH_RTSP="rtsp://user:pass@192.168.1.10/stream"
# repeat for each arm
python -m backend.main
```

## Test with MP4 video (visual preview)

Run YOLO + ROI detection on an MP4 with a live OpenCV preview window.
Metrics and ML predictions are also pushed to the frontend dashboard.

```bash
# Terminal 1 — backend (needed for frontend metrics)
python -m backend.main

# Terminal 2 — frontend
cd frontend && npm run dev

# Terminal 3 — preview
python scripts/preview_detection.py --video path/to/video.mp4 --camera JCT01_ARM_NORTH
```

**What the preview window shows:**
- Green bounding boxes = vehicles **inside** the ROI (tracked)
- Red bounding boxes = vehicles **outside** the ROI (ignored)
- Green polygon overlay = ROI region
- Yellow horizontal line = counting line
- Top-left HUD = frame count, detections, VPM, queue depth, LSTM score, anomaly score, alert level

**Draw a new ROI interactively:**

```bash
python scripts/preview_detection.py --video path/to/video.mp4 --draw-roi
```

Click polygon vertices around the target lane, press ENTER to confirm.

**Draw ROI and save it for the live pipeline:**

```bash
python scripts/preview_detection.py --video path/to/video.mp4 --draw-roi --save-roi JCT01_ARM_NORTH
```

**Headless mode (metrics to frontend only, no OpenCV window):**

```bash
python scripts/preview_detection.py --video path/to/video.mp4 --camera JCT01_ARM_NORTH --no-preview
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| SPACE | Pause / resume |
| q, ESC | Quit |
| s | Save current frame as PNG screenshot |

## Video with ROI saving (For videos)

```bash
# ROI Calibration
python scripts/preview_detection.py --video "C:\Users\PC\Desktop\koduvally data\site1\00006.mp4" --camera JCT01_ARM_NORTH --draw-roi --save-roi JCT01_ARM_NORTH

# Start Video for metrics
$env:DEMO_VIDEO_PATH="C:\Users\PC\Desktop\koduvally data\site1\00006.mp4"
python -m backend.main

```


## ROI calibration (for live cameras)

```bash
# Calibrate ROI for one camera
python scripts/setup_roi.py --camera JCT01_ARM_NORTH --source path/to/video.mp4

# Calibrate all cameras
python scripts/setup_roi.py --all

# Check which cameras have ROIs
python scripts/setup_roi.py --list

# Clear a camera's ROI
python scripts/setup_roi.py --clear JCT01_ARM_NORTH
```

Saved ROIs are stored in `data/roi_masks.json` and loaded automatically when the backend starts.

## Key URLs

| What | URL |
|------|-----|
| Dashboard | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

## Adjust thresholds at runtime (no restart)

```bash
curl -X POST http://localhost:8000/config \
     -H "Content-Type: application/json" \
     -d '{"key": "W3_PEAK_MULTIPLIER", "value": 1.5}'
```

Allowed keys: `W1_VOLUME_THRESHOLD`, `W2_VOLUME_THRESHOLD`, `W3_PEAK_MULTIPLIER`, `YOLO_CONFIDENCE_THRESHOLD`, `STOP_THRESHOLD`, `FRAME_RATE`, `ALERT_DEBOUNCE_MINUTES`, `ALERT_DEESCALATE_MINUTES`

## Docker

```bash
cd traffic_system
docker-compose up --build
```

## Extract metrics from a new video

```bash
python scripts/process_video_interactive.py --video path/to/video.mp4
# Click polygon ROI on first frame, ENTER to confirm, then wait
# Output: data/<video_name>_extracted.csv
```

## Retrain after new labelled data

```bash
# Add 'label' column (NORMAL / OFF_PEAK_JAM / PEAK_EXCESS) to extracted CSV first
python -m scripts.train_autoencoder --data data/my_labelled.csv
python -m scripts.train_lstm        --data data/my_labelled.csv
```
