# Quick Start (10 minutes)

## 1) Setup

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install ultralytics opencv-python numpy psutil
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy psutil
```

---

## 2) Run

### Option A: Webcam
```bash
python final_updated_moderate.py
```

### Option B: Video file
```bash
python final_updated_moderate.py path/to/video.mp4
```

Controls:
- **Q** quit
- **S** save screenshot

Outputs:
- `intrusion_report.json`
- optional `alert_frame_*.jpg`

---

## 3) Tune zones (most important)

Open `final_updated_moderate.py` → edit `DEFAULT_ZONES` polygons.

Rule of thumb:
- Restricted zones should tightly match the actual “no-entry” area.
- Monitored zones should match “watch-only” areas.

---

## 4) Baseline comparison (zone-only)

To reproduce a baseline run:
```python
ENABLE_BEHAVIOR_ANALYSIS = False
```

Then rerun and compare counts/severity in `intrusion_report.json`.

---

## 5) Common issues

- **Model download fails**: ensure internet access the first time (Ultralytics downloads weights).
- **Black window / no frames**: try a different camera index (0,1,2) or use a video file.
- **Slow FPS**: reduce input resolution, or use a smaller model variant if available on your machine.
