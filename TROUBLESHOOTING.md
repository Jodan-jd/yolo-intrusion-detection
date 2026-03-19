# Troubleshooting Guide

## Common Issues & Solutions for YOLO-Based Security System

---

## 🚀 Runtime Issues

### Issue: "No module named 'ultralytics'"
**When:** Running detection script  
**Cause:** Package not installed or virtual environment not activated  
**Solution:**
```bash
# Verify virtual environment is activated (should see (venv) in prompt)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall ultralytics
pip install --upgrade ultralytics

# Test import
python -c "from ultralytics import YOLO; print('OK')"
```

---

### Issue: "CUDA out of memory" or "RuntimeError: out of memory"
**When:** Processing large video  
**Cause:** GPU memory insufficient (if using GPU)  
**Solution:**
```bash
# Option 1: Use CPU instead
# Edit code: model = YOLO('yolov11n.pt').to('cpu')

# Option 2: Process smaller frames
# Add parameter: --fps-target 10

# Option 3: Close other GPU applications
# Free up GPU memory before running

# Option 4: Use smaller model
# Note: yolov11n is already smallest
```

---

### Issue: Low FPS (< 10)
**When:** Processing extremely slow  
**Cause:** CPU bottleneck, high resolution, or background processes  
**Solution:**

**Check 1: System Resources**
```bash
# Monitor CPU usage
# Windows: Task Manager → Performance
# macOS: Activity Monitor
# Linux: top or htop command

# Close unnecessary programs
```

**Check 2: Reduce Processing Load**
```bash
# Lower target FPS
python zone_behavior_detector.py --video input.mp4 --fps-target 10

# Lower input resolution (pre-process video)
# Or adjust YOLO confidence threshold
python zone_behavior_detector.py --video input.mp4 --confidence 0.5
```

**Check 3: Disable Visualization**
```bash
# Video display slows down processing
python zone_behavior_detector.py --video input.mp4 --visualize False
```

**Check 4: Check Performance Metrics**
- Normal: 15+ FPS on modern CPUs
- Slow: 5-10 FPS (consider optimization)
- Very Slow: < 5 FPS (major bottleneck)

---

### Issue: "Video file not found" or "Cannot open video"
**When:** Running detection  
**Cause:** Wrong video path or unsupported format  
**Solution:**
```bash
# Verify video file path
# Use absolute path if relative path fails
python zone_behavior_detector.py --video /full/path/to/video.mp4

# Supported formats: MP4, AVI, MOV, MKV, FLV, WMV
# Verify format with:
ffprobe video.mp4  # Or use media player

# If corrupted, re-encode video
# ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

---

### Issue: "No persons detected" / Empty results
**When:** Running on video but no alerts generated  
**Cause:** 
- No people in video
- Confidence threshold too high
- People too small/far from camera  
**Solution:**
```bash
# Check 1: Lower confidence threshold
python zone_behavior_detector.py --video input.mp4 --confidence 0.3

# Check 2: Verify video has people
# Watch video to confirm people are visible

# Check 3: Check console output
# Should show "Detected N persons in frame X"

# Check 4: Verify zone coordinates
# Edit config.json and verify zone polygons overlap video area
```

---

### Issue: "Behavior not detecting" (hand raise, stance, etc.)
**When:** System detects zones but not behaviors  
**Cause:** 
- Camera angle incompatible
- Person too distant
- Lighting issues
- Threshold too strict  
**Solution:**

**For Hand Raise Not Detecting:**
```python
# In code, adjust threshold (smaller = more sensitive)
HAND_RAISE_DISTANCE = 15  # Was 20, made stricter

# If still not working, person may be too far from camera
# Try closer video or different camera angle
```

**For Wide Stance/Kicking Not Detecting:**
```python
# Adjust ankle separation threshold
STANCE_WIDTH = 80  # Was 100, made stricter

# Check if person's legs are fully visible in frame
```

**For Crouch/Extreme Bend Not Detecting:**
```bash
# Known limitation: 0% detection with low-angle upward camera
# This is a camera angle issue, not a software bug

# Solutions:
# 1. Use camera at eye level or overhead
# 2. Or skip these behaviors for your setup
```

**Universal Fix:**
```python
# Temporarily disable confidence check for debugging
# Print keypoint positions to console
# Adjust thresholds based on actual measurements
```

---

### Issue: False Positives (too many CRITICAL alerts)
**When:** Getting alerts for normal activity  
**Cause:** Thresholds too loose  
**Solution:**

**Method 1: Increase Threshold**
```python
# Make behavior score requirement stricter
AGGRESSIVE_SCORE_THRESHOLD = 0.50  # Was 0.40, increased

# Or adjust individual behavior weights
# Reduce weights of sensitive behaviors
```

**Method 2: Adjust Individual Behavior Thresholds**
```python
# Make each behavior detection stricter
HAND_RAISE_DISTANCE = 25      # Was 20
KNEE_RAISE_HEIGHT = 60        # Was 50
STANCE_WIDTH = 120            # Was 100

# Higher values = less sensitive
```

**Method 3: Review & Remove Alert**
```bash
# Check alert JSON
# If it's legitimate (normal motion), thresholds need tuning
# Adjust for your specific facility/camera setup
```

---

### Issue: Missing Screenshots
**When:** No alert images saved  
**Cause:** 
- Alert severity too low
- Output folder doesn't exist
- Permission issue  
**Solution:**
```bash
# Verify output folder exists
mkdir -p output/alerts

# Check folder permissions
# Windows: Right-click → Properties → Security
# macOS/Linux: chmod 755 output/alerts

# Verify severity triggered
# Only CRITICAL and HIGH alerts generate screenshots
# MEDIUM alerts don't capture images

# Check output path in config
# Default: ./output/alerts/
```

---

## 📊 Output & Reporting Issues

### Issue: "intrusion_report.json not generated"
**When:** No JSON output file after processing  
**Cause:** 
- No alerts detected
- Output folder permission issue
- Script error  
**Solution:**
```bash
# Check console for errors
# End of output should show summary statistics

# Manually create output if needed
python zone_behavior_detector.py --video input.mp4 --output ./output

# Verify permissions
ls -la output/  # macOS/Linux
dir output\    # Windows
```

---

### Issue: "Statistics incomplete or missing"
**When:** Summary stats don't match actual alerts  
**Cause:** 
- Multi-run test (counts accumulate)
- Partial processing  
**Solution:**
```bash
# Delete previous output and rerun
rm -rf output/*        # macOS/Linux
rmdir /s output        # Windows

# Then run fresh
python zone_behavior_detector.py --video input.mp4 --output ./output
```

---

## 🔧 Configuration Issues

### Issue: "Zone not detecting properly"
**When:** People in zone but no alerts  
**Cause:** Zone polygon coordinates incorrect  
**Solution:**
```bash
# Verify zone coordinates in config.json
# Coordinates should be: [[x1,y1], [x2,y2], [x3,y3], ...]
# Where x=horizontal (0=left), y=vertical (0=top)

# Check video resolution first
# Zone coords must match video dimensions

# Use calibration tool (if available)
# Click points on video to generate polygon

# Or manually verify by displaying overlay
```

**Example Zone Fix:**
```json
{
  "polygon": [[500, 505], [595, 361], [757, 395], [594, 520]],
  "type": "restricted"
}
```

---

### Issue: "Invalid configuration JSON"
**When:** Reading config.json fails  
**Cause:** JSON syntax error  
**Solution:**
```bash
# Validate JSON syntax (use online validator or command)
python -m json.tool config.json

# Common errors:
# - Missing commas between elements
# - Trailing commas in arrays
# - Mismatched brackets

# Example correct JSON:
{
  "zones": [
    {
      "name": "Zone1",
      "polygon": [[x1,y1], [x2,y2]],
      "type": "monitored"
    }
  ]
}
```

---

## 🖥️ System & Environment Issues

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**When:** Running script  
**Cause:** OpenCV not installed  
**Solution:**
```bash
pip install opencv-python
# Or if that fails:
pip install opencv-contrib-python

# Test
python -c "import cv2; print(cv2.__version__)"
```

---

### Issue: "ImportError: DLL load failed" (Windows)
**When:** Running on Windows  
**Cause:** Missing Visual C++ runtime  
**Solution:**
```bash
# Download Visual C++ Redistributable for Visual Studio 2022
# https://support.microsoft.com/en-us/help/2977003

# Or reinstall PyTorch
pip install --force-reinstall torch

# Or use conda instead of pip
conda install pytorch torchvision torchaudio -c pytorch
```

---

### Issue: Virtual Environment Activation Fails
**When:** Trying to activate venv  
**Cause:** Permission issue or path issue  
**Solution:**

**Windows PowerShell:**
```powershell
# Enable execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.\venv\Scripts\Activate.ps1
```

**macOS/Linux Permission:**
```bash
# Make activate script executable
chmod +x venv/bin/activate

# Then activate
source venv/bin/activate
```

---

### Issue: "Python version too old"
**When:** Running script  
**Cause:** Python < 3.10  
**Solution:**
```bash
# Check version
python --version

# If 3.9 or lower, upgrade
# Windows: Download Python 3.10+ from python.org
# macOS: brew install python@3.10
# Linux: sudo apt install python3.10

# Or create new venv with specific Python
python3.10 -m venv venv
source venv/bin/activate
```

---

## 🔄 Model Download Issues

### Issue: "Models downloading very slowly"
**When:** First run takes very long  
**Cause:** Large files, slow internet  
**Solution:**
```bash
# Pre-download models manually
python -c "from ultralytics import YOLO; YOLO('yolov11n.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov11n-pose.pt')"

# Or download to specific directory
export YOLO_HOME=/path/to/models
python zone_behavior_detector.py --video input.mp4
```

---

### Issue: "Model download fails or corrupted"
**When:** Can't load model  
**Cause:** Interrupted download, corrupted file  
**Solution:**
```bash
# Delete cached models
rm -rf ~/.config/Ultralytics  # macOS/Linux
rmdir /s %APPDATA%\Ultralytics  # Windows

# Rerun to re-download
python zone_behavior_detector.py --video input.mp4

# Or manually place models in correct directory
# ~/.config/Ultralytics/weights/
```

---

## 🎥 Video Format Issues

### Issue: "Unsupported video format"
**When:** Video won't open  
**Cause:** Format not supported  
**Solution:**
```bash
# Supported: MP4, AVI, MOV, MKV, FLV, WMV

# Convert using ffmpeg (install if needed)
ffmpeg -i input.mov -c:v libx264 -c:a aac -q:v 5 output.mp4

# Or use any video converter
```

---

### Issue: "Video plays but slow processing"
**When:** Video works but FPS low  
**Cause:** High resolution or bitrate  
**Solution:**
```bash
# Option 1: Reduce resolution with ffmpeg
ffmpeg -i input.mp4 -s 640x480 output.mp4

# Option 2: Lower bitrate
ffmpeg -i input.mp4 -b:v 2M output.mp4

# Option 3: Lower frame rate before processing
ffmpeg -i input.mp4 -r 15 output.mp4
```

---

## 📈 Performance Tuning

### Issue: "Need faster processing"
**Cause:** CPU bottleneck  
**Solution:**

**Method 1: GPU Acceleration**
```bash
# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Expected: 2-3× faster on GPU
```

**Method 2: Lower Processing Load**
```bash
# Process fewer frames
python zone_behavior_detector.py --video input.mp4 --fps-target 10

# Lower resolution
# Pre-process video to smaller size
```

**Method 3: Disable Features**
```bash
# Disable visualization (slow on display)
python zone_behavior_detector.py --video input.mp4 --visualize False

# Reduces overhead by ~30%
```

---

## 📝 Debugging Steps

**When nothing works, follow this systematic approach:**

1. **Check Python Setup**
   ```bash
   python --version          # Should be 3.10+
   which python              # Verify correct path
   pip list | grep ultralytics  # Check packages
   ```

2. **Verify Installation**
   ```bash
   python -c "import torch, cv2, ultralytics; print('OK')"
   ```

3. **Test with Sample**
   ```bash
   # Use a known-working small video
   python zone_behavior_detector.py --video test_small.mp4
   ```

4. **Check Errors**
   ```bash
   # Copy full error message
   # Search error in this guide
   # Or in README.md troubleshooting
   ```

5. **Review Logs**
   ```bash
   # Check console output for "ERROR" or "Exception"
   # Last few lines usually show problem
   ```

---

## 📞 When to Contact Support

If issue persists after troubleshooting:
1. Document exact error message
2. Include Python version and OS
3. List installed package versions: `pip list`
4. Note video format and size
5. Check INSTALLATION.md and README.md first

---

## 🆘 Emergency Reset

If everything fails, clean reinstall:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv              # macOS/Linux
rmdir /s venv           # Windows

# Remove cache
rm -rf ~/.cache/pip     # macOS/Linux
rm -rf %APPDATA%\.cache # Windows

# Delete models
rm -rf ~/.config/Ultralytics

# Reinstall from scratch
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt

# Re-download models
python -c "from ultralytics import YOLO; YOLO('yolov11n.pt'); YOLO('yolov11n-pose.pt')"

# Try again
python zone_behavior_detector.py --video input.mp4
```

---

**Still having issues?** Review the complete [README.md](README.md) and [INSTALLATION.md](INSTALLATION.md) files.
