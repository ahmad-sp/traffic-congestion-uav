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
