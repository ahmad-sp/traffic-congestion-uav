"""
Central configuration for the AI Traffic Congestion Detection & Early Warning System.

ALL configurable parameters live here — no hardcoded thresholds anywhere else.
Import this module wherever you need a parameter value.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models_saved"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
LOG_DIR = PROJECT_ROOT / "logs"

for _dir in (DATA_DIR, MODEL_DIR, EVIDENCE_DIR, SYNTHETIC_DATA_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'traffic.db'}")

# ─────────────────────────────────────────────
# FRAME INGESTION
# ─────────────────────────────────────────────
FRAME_RATE = int(os.getenv("FRAME_RATE", "5"))          # FPS to process
DEMO_VIDEO_PATH = os.getenv("DEMO_VIDEO_PATH", "")      # fallback .mp4 when no RTSP
FRAME_QUEUE_MAXSIZE = 128                                 # per-camera frame buffer

# ─────────────────────────────────────────────
# YOLO DETECTION
# ─────────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.45"))
# COCO class IDs to keep: car=2, motorcycle=3, bus=5, truck=7
YOLO_TARGET_CLASSES = [2, 3, 5, 7]
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")  # "cpu", "cuda:0", etc.

# ─────────────────────────────────────────────
# TRACKING (ByteTrack)
# ─────────────────────────────────────────────
TRACK_HIGH_THRESH = 0.5        # detection confidence for first association
TRACK_LOW_THRESH = 0.1         # detection confidence for second association
TRACK_MATCH_THRESH = 0.8       # IoU threshold
TRACK_BUFFER = 30              # frames to keep lost tracks (= 6 seconds at 5 FPS)
TRACK_FRAME_RATE = FRAME_RATE

# ─────────────────────────────────────────────
# MOTION & QUEUE ANALYSIS
# ─────────────────────────────────────────────
STOP_THRESHOLD = float(os.getenv("STOP_THRESHOLD", "2.0"))   # px/frame — below = stopped
STOP_CONSECUTIVE_FRAMES = 3                                    # must be stopped this many consecutive frames
APPROACH_THRESHOLD = float(os.getenv("APPROACH_THRESHOLD", "5.0"))  # bbox_area_delta above this = approaching

# Zone boundaries (fraction of frame height)
NEAR_ZONE_Y_FRACTION = 0.6    # centroid_y > this = near zone (close to camera)
FAR_ZONE_Y_FRACTION = 0.4     # centroid_y < this = far zone (far from camera)
FAR_ZONE_UPPER_Y_RATIO = 0.4  # upper boundary for far-zone bbox growth rate calculation

# Counting line (fraction of frame height)
COUNTING_LINE_Y_FRACTION = float(os.getenv("COUNTING_LINE_Y", "0.5"))

# Queue depth: near-zone stopped count sustained over this many seconds
QUEUE_DEPTH_SUSTAIN_SECONDS = 30

# ─────────────────────────────────────────────
# ML — LSTM CONGESTION FORECASTER
# ─────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 60       # 60 timesteps (minutes) input window
LSTM_NUM_FEATURES = 11          # features per timestep (10 original + mean_bbox_growth_rate)
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_FC_HIDDEN = 64

LSTM_CONGESTION_THRESHOLD = 0.7   # score >= this = congested

# Extreme congestion early detection (dual-head LSTM extension)
EXTREME_CONGESTION_FORECAST_THRESHOLD = float(os.getenv("EXTREME_CONGESTION_FORECAST_THRESHOLD", "0.65"))
EXTREME_CONGESTION_FORECAST_WINDOW = 10  # minutes lookahead for extreme risk label
LSTM_LOSS_WEIGHT_CONGESTION = 0.6
LSTM_LOSS_WEIGHT_EXTREME = 0.4

LSTM_TRAIN_LR = 1e-3
LSTM_TRAIN_BATCH_SIZE = 32
LSTM_TRAIN_EPOCHS = 50

LSTM_INFERENCE_INTERVAL_SECONDS = 60   # run every 60s

# ─────────────────────────────────────────────
# ML — AUTOENCODER ANOMALY DETECTOR
# ─────────────────────────────────────────────
AE_INPUT_DIM = 10
AE_ENCODER_DIMS = [32, 16, 8]
AE_DECODER_DIMS = [16, 32, 10]  # mirror of encoder output→input

AE_TRAIN_LR = 1e-3
AE_TRAIN_BATCH_SIZE = 32
AE_TRAIN_EPOCHS = 50

AE_ANOMALY_PERCENTILE = 95     # threshold = 95th percentile of training recon errors
AE_INFERENCE_INTERVAL_SECONDS = 60

# ─────────────────────────────────────────────
# WARRANT ENGINE THRESHOLDS
# ─────────────────────────────────────────────
W1_VOLUME_THRESHOLD = float(os.getenv("W1_VOLUME_THRESHOLD", "12"))   # VPM for 8-hour sustained
W2_VOLUME_THRESHOLD = float(os.getenv("W2_VOLUME_THRESHOLD", "15"))   # VPM for 4-hour consecutive
W3_PEAK_MULTIPLIER = float(os.getenv("W3_PEAK_MULTIPLIER", "1.40"))   # current VPM > baseline_85th * this

# Hours lookback for warrant checks
W1_HOURS = 8
W2_HOURS = 4

# ─────────────────────────────────────────────
# ALERT SYSTEM
# ─────────────────────────────────────────────
ALERT_DEBOUNCE_MINUTES = 5        # don't re-issue same level within this window
ALERT_DEESCALATE_MINUTES = 2      # consecutive clean minutes before de-escalation

# ─────────────────────────────────────────────
# DRONE TRIGGER
# ─────────────────────────────────────────────
DRONE_WEBHOOK_URL = os.getenv("DRONE_WEBHOOK_URL", "")  # empty = mock/log-only
EVIDENCE_CLIP_SECONDS = 30       # seconds of video to save before RED alert

# ─────────────────────────────────────────────
# PEAK HOURS (used for Case A vs Case B classification)
# ─────────────────────────────────────────────
# Defined as (start_hour, end_hour) in 24h format — half-open intervals
PEAK_PERIODS = [
    (7, 9),     # morning peak 07:00–09:00
    (17, 19),   # evening peak 17:00–19:00
]

def is_peak_hour(hour: int) -> bool:
    """Check if a given hour (0–23) falls within any peak period."""
    return any(start <= hour < end for start, end in PEAK_PERIODS)

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────
SYNTH_DAYS = 30                    # days to generate
SYNTH_DEFAULT_ARMS = 2             # default number of arms

# VPM ranges by period (min, max)
SYNTH_MORNING_PEAK_VPM = (18, 25)
SYNTH_MORNING_PEAK_NOISE_STD = 3
SYNTH_MORNING_PEAK_HOURS = (7.5, 9.0)   # 07:30–09:00

SYNTH_EVENING_PEAK_VPM = (20, 28)
SYNTH_EVENING_PEAK_NOISE_STD = 4
SYNTH_EVENING_PEAK_HOURS = (17.0, 19.0)

SYNTH_DAYTIME_VPM = (8, 14)
SYNTH_DAYTIME_HOURS = (9.0, 17.0)

SYNTH_NIGHT_VPM = (1, 5)
SYNTH_NIGHT_HOURS_START = 22.0
SYNTH_NIGHT_HOURS_END = 6.0

SYNTH_WEEKEND_SCALE = 0.6

# Abnormal event injection
SYNTH_OFFPEAK_JAM_VPM = (0, 2)
SYNTH_OFFPEAK_JAM_DURATION_MIN = (20, 40)
SYNTH_OFFPEAK_JAM_QUEUE_DEPTH = (8, 15)
SYNTH_OFFPEAK_JAM_STOPPED_RATIO = 0.85

SYNTH_PEAK_EXCESS_VPM = (35, 45)
SYNTH_PEAK_EXCESS_QUEUE_DEPTH = (10, 20)
SYNTH_PEAK_EXCESS_DURATION_MIN = (60, 120)

# ─────────────────────────────────────────────
# FASTAPI / SERVER
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

WS_BROADCAST_INTERVAL_SECONDS = 5   # WebSocket push frequency

# ─────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
WS_BASE_URL = os.getenv("WS_BASE_URL", f"ws://localhost:{API_PORT}")

# ─────────────────────────────────────────────
# JUNCTION & ARM DEFINITIONS
# ─────────────────────────────────────────────
# Each junction has a type (L/T/+), a set of arms, and GPS coordinates.
# Camera ID = {junction_id}_{arm_id}
# Expand or modify this dict for your deployment.

JUNCTIONS = {
    "JCT01": {
        "name": "Main Street / Oak Avenue",
        "type": "+",                          # crossroads
        "arms": {
            "ARM_NORTH": {
                "name": "Main St Northbound Approach",
                "gps_lat": 51.5074,
                "gps_lon": -0.1278,
                "rtsp_url": os.getenv("JCT01_ARM_NORTH_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
            "ARM_SOUTH": {
                "name": "Main St Southbound Approach",
                "gps_lat": 51.5064,
                "gps_lon": -0.1278,
                "rtsp_url": os.getenv("JCT01_ARM_SOUTH_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
            "ARM_EAST": {
                "name": "Oak Ave Eastbound Approach",
                "gps_lat": 51.5069,
                "gps_lon": -0.1268,
                "rtsp_url": os.getenv("JCT01_ARM_EAST_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
            "ARM_WEST": {
                "name": "Oak Ave Westbound Approach",
                "gps_lat": 51.5069,
                "gps_lon": -0.1288,
                "rtsp_url": os.getenv("JCT01_ARM_WEST_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
        },
    },
    "JCT02": {
        "name": "High Road / Park Lane",
        "type": "T",                          # T-junction
        "arms": {
            "ARM_NORTH": {
                "name": "High Rd Northbound Approach",
                "gps_lat": 51.5120,
                "gps_lon": -0.1400,
                "rtsp_url": os.getenv("JCT02_ARM_NORTH_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
            "ARM_SOUTH": {
                "name": "High Rd Southbound Approach",
                "gps_lat": 51.5110,
                "gps_lon": -0.1400,
                "rtsp_url": os.getenv("JCT02_ARM_SOUTH_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
            "ARM_EAST": {
                "name": "Park Ln Eastbound Approach",
                "gps_lat": 51.5115,
                "gps_lon": -0.1390,
                "rtsp_url": os.getenv("JCT02_ARM_EAST_RTSP", ""),
                "counting_line_y": COUNTING_LINE_Y_FRACTION,
            },
        },
    },
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# ─────────────────────────────────────────────
# HELPER — enumerate all camera IDs
# ─────────────────────────────────────────────
def get_all_camera_ids() -> list[str]:
    """Return a list of all camera IDs in the format junction_id_arm_id."""
    ids = []
    for jid, jdata in JUNCTIONS.items():
        for aid in jdata["arms"]:
            ids.append(f"{jid}_{aid}")
    return ids


def get_arm_config(junction_id: str, arm_id: str) -> dict:
    """Return the config dict for a specific arm, or raise KeyError."""
    return JUNCTIONS[junction_id]["arms"][arm_id]
