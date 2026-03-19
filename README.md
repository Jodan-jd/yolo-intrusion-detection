# 🎯 Context-Aware Intrusion Detection using YOLO

## 📌 Overview

This project presents a real-time intrusion detection system that combines **computer vision + behavioral analysis** to reduce alert fatigue in physical security systems.

Traditional systems treat all intrusions equally. This system introduces **context-aware detection** by analyzing:

* **WHERE** → zone-based intrusion detection
* **HOW** → pose-based behavior analysis
* **SO WHAT** → severity-based alert prioritization

---

## 🚀 Key Results

* ✅ **100% Zone Detection Accuracy**
* ✅ **95% Reduction in False Alerts**
* ✅ **97% Reduction in Security Operator Workload**
* ⚡ **15 FPS Real-time Processing (CPU-only)**

---

## 🧠 System Design

### 📍 Spatial Layer (WHERE)

* Polygon-based zone detection
* Ray-casting algorithm
* Supports:

  * Restricted zones
  * Monitored zones

---

### 🧍 Behavioral Layer (HOW)

* YOLO Pose (17 keypoints)
* Detects:

  * Hand raise
  * Knee raise / kicking
  * Wide stance
  * Crouching
  * Extreme bending

---

### ⚡ Decision Layer (SO WHAT)

Combines location + behavior:

* Restricted + aggressive → **CRITICAL**
* Restricted + normal → **HIGH**
* Monitored → **MEDIUM**

---

## 🛠️ Tech Stack

* Python
* YOLO (Ultralytics)
* OpenCV
* NumPy

---

## ▶️ Quick Start (5–10 mins)

```bash
python final_updated_moderate.py
```

OR

```bash
python final_updated_moderate.py path/to/video.mp4
```

Controls:

* Q → Quit
* S → Save screenshot

Outputs:

* intrusion_report.json
* alert_frame_*.jpg

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python numpy psutil
```

---

## 📂 Project Structure

```bash
yolo-intrusion-detection/
├── docs/
│   ├── project-report.pdf
│   ├── presentation.pptx
├── src/
├── QUICKSTART.md
├── TROUBLESHOOTING.md
└── requirements.txt
```

---

## 🎥 Demo

👉 (Add your video link here — YouTube or Drive)

---

## 📄 Documentation

* 📘 Full Report → `[docs/project-report.pdf](https://github.com/Jodan-jd/yolo-intrusion-detection/blob/main/docs/Project%20-%20report.pdf)`
* ⚡ Quick Start → QUICKSTART.md
* 🛠️ Troubleshooting → TROUBLESHOOTING.md

---

## ⚙️ Configuration

Modify in `final_updated_moderate.py`:

* Zone polygons
* Detection thresholds
* Behavior weights

Example:

```python
ENABLE_BEHAVIOR_ANALYSIS = True
```

---

## 📊 Key Insight

Traditional systems generate excessive alerts.

👉 By combining:

* Spatial context
* Behavioral analysis

This system:

* Prioritizes real threats
* Reduces operator workload
* Improves response efficiency

---

## ⚠️ Limitations

* Single-person testing
* Limited environment validation
* Some behaviors need tuning (crouch, knee raise)

---

## 🧠 Perspective

This project reflects my interest in:

* AI-driven security systems
* Real-time decision systems
* Context-aware intelligence
* Reducing noise in large-scale systems

---

