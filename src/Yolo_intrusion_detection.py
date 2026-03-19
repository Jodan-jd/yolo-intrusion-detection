#!/usr/bin/env python3
"""
YOLOv11 ZONE-BASED INTRUSION DETECTION WITH POSE-BASED BEHAVIOR ANALYSIS
Physical Security and Intrusion Detection System

FINAL DUAL-MODE VERSION:
✅ Walking not detected as crouching
✅ Knee raise (kicking) detection: ABSOLUTE + RATIO (distance-invariant)
✅ Hand raise detection: ABSOLUTE + RATIO (distance-invariant)
✅ Proper thresholds to avoid false positives
✅ Frame smoothing for stability
✅ Screenshot saving ('S' key)
✅ Intrusion report saved as JSON (intrusion_report.json)
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import json
import sys
import psutil
import time

# ⚙️ ========== THRESHOLD CONFIGURATION ==========

# Enable/disable behavior analysis for comparison
ENABLE_BEHAVIOR_ANALYSIS = True    # Set to False for baseline zone-only detection

DETECTION_CONF = 0.4               # YOLO detection confidence

# Absolute pixel thresholds
HAND_RAISE_DISTANCE = 20           # px above shoulder (strict near camera)
CROUCH_DISTANCE = 80               # knee below hip (strict)
STANCE_WIDTH = 100                 # px between ankles (wide stance)
EXTREME_BEND_DISTANCE = 60         # nose below hips (extreme bending)
KNEE_RAISE_HEIGHT = 50             # px - knee raised ABOVE hip (kicking motion)

# Relative thresholds (as fraction of body height) – used together with absolute thresholds
HAND_RAISE_DISTANCE_RATIO = 0.10   # 15% of body height above shoulder
KNEE_RAISE_HEIGHT_RATIO = 0.15     # 25% of body height above hip

# Behavior weights
HAND_RAISE_WEIGHT = 0.80           # Strong indicator
CROUCH_WEIGHT = 0.35               # Crouch indicator
STANCE_WEIGHT = 0.60               # Wide stance indicator
EXTREME_BEND_WEIGHT = 0.25
KNEE_RAISE_WEIGHT = 0.90           # Kicking is clearly aggressive

# Behavior score thresholds for alerts
AGGRESSIVE_SCORE_THRESHOLD = 0.40  # Above this → aggressive
WARNING_SCORE_THRESHOLD = 0.30     # Medium behavior

# Frame smoothing
BEHAVIOR_MEMORY_FRAMES = 2         # Average score over N frames

# Zone definitions (example – adjust for your scene)
DEFAULT_ZONES = [
    {
        'name': 'Restricted Area 1',
        'polygon': [(50, 50), (300, 50), (300, 300), (50, 300)],
        'type': 'restricted',
        'color': (0, 0, 255)  # Red
    },
    {
        'name': 'Monitored Zone',
        'polygon': [(400, 100), (600, 100), (600, 400), (400, 400)],
        'type': 'monitored',
        'color': (0, 165, 255)  # Orange
    }
]


class ZoneBehaviorDetector:
    def __init__(self, zones=None):
        print("🔥 Loading YOLO detection model...")
        try:
            self.model = YOLO('yolo11n.pt')
            print("✓ Detection model loaded (yolo11n.pt)\n")
        except Exception as e:
            print(f"⚠ Error loading yolo11n.pt: {e}")
            print("Trying yolov8n.pt...")
            self.model = YOLO('yolov8n.pt')

        print("🔥 Loading YOLO pose model...")
        try:
            self.pose_model = YOLO('yolo11n-pose.pt')
            print("✓ Pose model loaded (yolo11n-pose.pt)\n")
        except Exception as e:
            print(f"⚠ Error loading yolo11n-pose.pt: {e}")
            print("Trying yolov8n-pose.pt...")
            try:
                self.pose_model = YOLO('yolov8n-pose.pt')
                print("✓ Pose model loaded (yolov8n-pose.pt)\n")
            except Exception:
                print("⚠ Pose model not available. Behavioral analysis disabled.")
                self.pose_model = None

        # Use provided zones or defaults
        self.zones = zones if zones is not None else DEFAULT_ZONES

        self.frame_count = 0
        self.intrusion_count = 0
        self.alerts = []
        self.start_time = time.time()

        # Performance tracking
        self.frame_times = []
        self.cpu_usage = []
        self.memory_usage = []

        # Simple person tracking
        self.tracked_persons = {}
        self.next_track_id = 0

        # Behavior smoothing
        self.behavior_history = {}
        self.screenshot_count = 0

        print("⚙️ CONFIGURATION LOADED:")
        print(f"  Detection Confidence: {DETECTION_CONF}")
        print(f"  Behavior Analysis: {'ENABLED' if ENABLE_BEHAVIOR_ANALYSIS else 'DISABLED'}")
        print(f"  Aggressive Threshold: {AGGRESSIVE_SCORE_THRESHOLD}")
        print(f"  Warning Threshold: {WARNING_SCORE_THRESHOLD}")
        print(f"  Frame smoothing: {BEHAVIOR_MEMORY_FRAMES} frames")
        print(f"  Zones configured: {len(self.zones)}")
        for zone in self.zones:
            print(f"    • {zone['name']} ({zone['type']})")
        print()

    # ---------- SYSTEM STATS ----------

    def get_stats(self):
        """Get system stats"""
        try:
            cpu = psutil.cpu_percent(interval=0.01)
            mem = psutil.virtual_memory()
            self.cpu_usage.append(cpu)
            self.memory_usage.append(mem.percent)
            return cpu, mem.percent
        except Exception:
            return 0, 0

    # ---------- GEOMETRY / ZONES ----------

    @staticmethod
    def point_in_polygon(point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def check_zone_intrusion(self, bbox, class_name):
        """Check if detected object violates any zones"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        violations = []
        for zone in self.zones:
            if self.point_in_polygon((cx, cy), zone['polygon']):
                if zone['type'] == 'restricted':
                    if class_name == 'person':
                        violations.append({
                            'zone': zone['name'],
                            'zone_type': 'restricted',
                            'type': 'UNAUTHORIZED_ENTRY',
                            'severity': 'HIGH'
                        })
                elif zone['type'] == 'monitored':
                    violations.append({
                        'zone': zone['name'],
                        'zone_type': 'monitored',
                        'type': 'MONITORED_ACTIVITY',
                        'severity': 'MEDIUM'
                    })
        return violations

    # ---------- TRACKING ----------

    @staticmethod
    def get_distance(pos1, pos2):
        """Calculate Euclidean distance"""
        return float(np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))

    def simple_track(self, detections, prev_tracks, max_distance=50):
        """Simple tracking: match detections to previous tracks by proximity"""
        new_tracks = {}
        used_track_ids = set()

        for detection in detections:
            bbox = detection['bbox']
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            position = (cx, cy)

            best_match_id = None
            best_distance = max_distance

            for track_id, track_data in prev_tracks.items():
                if track_id in used_track_ids:
                    continue
                dist = self.get_distance(position, track_data['position'])
                if dist < best_distance:
                    best_distance = dist
                    best_match_id = track_id

            if best_match_id is not None:
                track_id = best_match_id
                used_track_ids.add(track_id)
            else:
                track_id = self.next_track_id
                self.next_track_id += 1

            new_tracks[track_id] = {
                'bbox': bbox,
                'position': position,
                'class': detection['class'],
                'confidence': detection['confidence'],
                'last_seen': self.frame_count
            }

        return new_tracks

    def smooth_behavior_score(self, person_id, raw_score):
        """Smooth behavior score over multiple frames to reduce jitter"""
        if person_id not in self.behavior_history:
            self.behavior_history[person_id] = deque(maxlen=BEHAVIOR_MEMORY_FRAMES)

        self.behavior_history[person_id].append(raw_score)
        smoothed_score = float(np.mean(self.behavior_history[person_id]))
        return smoothed_score

    # ---------- POSE / BEHAVIOR ANALYSIS (DUAL-MODE) ----------

    def analyze_keypoints(self, keypoints):
        """
        Analyze pose keypoints for behaviors.

        - Knee raise detection uses ABSOLUTE + RATIO (body-height normalized)
        - Hand raise detection uses ABSOLUTE + RATIO
        - Crouching requires BOTH legs bent symmetrically
        - Walking no longer triggers false crouching

        Returns: (score, behaviors_list)
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0, []

        k = keypoints
        behaviors = []
        score = 0.0

        try:
            # COCO-style index mapping for pose
            nose          = k[0]
            left_shoulder = k[5]
            right_shoulder = k[6]
            left_wrist    = k[9]
            right_wrist   = k[10]
            left_hip      = k[11]
            right_hip     = k[12]
            left_knee     = k[13]
            right_knee    = k[14]
            left_ankle    = k[15]
            right_ankle   = k[16]

            # Approximate body height (top to bottom in image)
            ys = [
                p[1] for p in [
                    nose, left_shoulder, right_shoulder,
                    left_hip, right_hip, left_ankle, right_ankle
                ] if p is not None
            ]
            body_height = None
            if ys:
                body_height = max(ys) - min(ys)

            # Average hip position (for extreme bend)
            avg_hip_y = None
            if left_hip is not None and right_hip is not None:
                avg_hip_y = (left_hip[1] + right_hip[1]) / 2

            # 1. KNEE RAISED (kicking/attacking) - DETECT FIRST
            knee_raised_left = False
            knee_raised_right = False

            if body_height is not None and body_height > 0:
                if left_knee is not None and left_hip is not None:
                    knee_above_hip_px = left_hip[1] - left_knee[1]
                    knee_above_hip_ratio = knee_above_hip_px / body_height
                    if (knee_above_hip_px > KNEE_RAISE_HEIGHT or
                        knee_above_hip_ratio > KNEE_RAISE_HEIGHT_RATIO):
                        score += KNEE_RAISE_WEIGHT
                        behaviors.append("LEFT_KNEE_RAISED")
                        knee_raised_left = True

                if right_knee is not None and right_hip is not None:
                    knee_above_hip_px = right_hip[1] - right_knee[1]
                    knee_above_hip_ratio = knee_above_hip_px / body_height
                    if (knee_above_hip_px > KNEE_RAISE_HEIGHT or
                        knee_above_hip_ratio > KNEE_RAISE_HEIGHT_RATIO):
                        score += KNEE_RAISE_WEIGHT
                        behaviors.append("RIGHT_KNEE_RAISED")
                        knee_raised_right = True

            # 2. HANDS RAISED (threatening gesture)
            hands_raised_count = 0

            if body_height is not None and body_height > 0:
                if left_wrist is not None and left_shoulder is not None:
                    hand_above_shoulder_px = left_shoulder[1] - left_wrist[1]
                    hand_above_shoulder_ratio = hand_above_shoulder_px / body_height
                    if (hand_above_shoulder_px > HAND_RAISE_DISTANCE or
                        hand_above_shoulder_ratio > HAND_RAISE_DISTANCE_RATIO):
                        hands_raised_count += 1
                        behaviors.append("LEFT_ARM_RAISED")

                if right_wrist is not None and right_shoulder is not None:
                    hand_above_shoulder_px = right_shoulder[1] - right_wrist[1]
                    hand_above_shoulder_ratio = hand_above_shoulder_px / body_height
                    if (hand_above_shoulder_px > HAND_RAISE_DISTANCE or
                        hand_above_shoulder_ratio > HAND_RAISE_DISTANCE_RATIO):
                        hands_raised_count += 1
                        behaviors.append("RIGHT_ARM_RAISED")

            # Scale weight based on number of hands raised
            if hands_raised_count > 0:
                score += HAND_RAISE_WEIGHT * (hands_raised_count / 2.0)

            # 3. CROUCHING - STRICT CHECK (only if NOT kicking)
            if not (knee_raised_left or knee_raised_right):
                if (left_knee is not None and left_hip is not None and left_ankle is not None and
                    right_knee is not None and right_hip is not None and right_ankle is not None):

                    left_knee_hip = left_knee[1] - left_hip[1]
                    right_knee_hip = right_knee[1] - right_hip[1]

                    both_knees_bent = (
                        left_knee_hip > CROUCH_DISTANCE and
                        right_knee_hip > CROUCH_DISTANCE
                    )

                    ankles_below_knees = (
                        left_ankle[1] > left_knee[1] and
                        right_ankle[1] > right_knee[1]
                    )

                    similar_knee_bend = abs(left_knee_hip - right_knee_hip) < 50

                    if both_knees_bent and ankles_below_knees and similar_knee_bend:
                        score += CROUCH_WEIGHT
                        behaviors.append("CROUCHING")

            # 4. WIDE STANCE (aggressive posture)
            if left_ankle is not None and right_ankle is not None:
                stance = abs(left_ankle[0] - right_ankle[0])
                if stance > STANCE_WIDTH:
                    score += STANCE_WEIGHT
                    behaviors.append("WIDE_STANCE")

            # 5. EXTREME BODY ANGLE (bending/ducking)
            if nose is not None and avg_hip_y is not None:
                nose_below_hips = nose[1] - avg_hip_y
                if nose_below_hips > EXTREME_BEND_DISTANCE:
                    score += EXTREME_BEND_WEIGHT
                    behaviors.append("EXTREME_BEND")

        except Exception:
            # Silent fail - don't crash on pose estimation errors
            pass

        score = min(score, 1.0)
        return score, behaviors

    def get_person_pose(self, frame, bbox):
        """Run pose model on cropped person and return keypoints."""
        if self.pose_model is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        results = self.pose_model.predict(person_crop, conf=0.3, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None

        kpts = results[0].keypoints[0].xy[0]
        if hasattr(kpts, 'cpu'):
            kpts = kpts.cpu().numpy()

        # Convert keypoints back to original frame coordinates
        kpts[:, 0] += x1
        kpts[:, 1] += y1
        return kpts

    # ---------- DRAW ZONES ----------

    def draw_zones(self, frame):
        """Draw zone overlays on frame"""
        overlay = frame.copy()
        for zone in self.zones:
            pts = np.array(zone['polygon'], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], zone['color'])
            cv2.polylines(frame, [pts], True, zone['color'], 2)
            center_x = int(np.mean([p[0] for p in zone['polygon']]))
            center_y = int(np.mean([p[1] for p in zone['polygon']]))
            cv2.putText(frame, zone['name'], (center_x - 50, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame

    def save_screenshot(self, frame, frame_num):
        """Save annotated frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_frame_{timestamp}_f{frame_num}.jpg"
        cv2.imwrite(filename, frame)
        self.screenshot_count += 1
        print(f"\n📸 Screenshot saved: {filename}")
        print(f"   Total screenshots: {self.screenshot_count}\n")

    # ---------- MAIN FRAME PROCESSING ----------

    def process_frame(self, frame):
        """Process single frame"""
        self.frame_count += 1
        frame_start = time.time()

        cpu, mem = self.get_stats()
        annotated = self.draw_zones(frame.copy())

        # Detect persons
        results = self.model.predict(frame, conf=DETECTION_CONF, verbose=False)
        result = results[0]

        person_detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                if class_name != 'person':
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf
                }
                person_detections.append(detection)

        # Track persons
        self.tracked_persons = self.simple_track(person_detections, self.tracked_persons)

        # Process each tracked person
        for person_id, person_data in self.tracked_persons.items():
            bbox = person_data['bbox']
            violations = self.check_zone_intrusion(bbox, 'person')

            behavior_score = 0.0
            behaviors = []
            behavior_label = "NORMAL"

            # If person is in a zone, analyze their behavior (if enabled)
            if violations and ENABLE_BEHAVIOR_ANALYSIS:
                keypoints = self.get_person_pose(frame, bbox)
                raw_score, behaviors = self.analyze_keypoints(keypoints)

                # Use smoothed score to reduce jitter
                behavior_score = self.smooth_behavior_score(person_id, raw_score)

                # Classify behavior
                if behavior_score >= AGGRESSIVE_SCORE_THRESHOLD:
                    behavior_label = "AGGRESSIVE_POSE"
                elif behavior_score >= WARNING_SCORE_THRESHOLD:
                    behavior_label = "SUSPICIOUS_POSE"
                else:
                    behavior_label = "NORMAL_POSE"

                # Force AGGRESSIVE if leg raised in monitored or restricted zone
                if any(b in behaviors for b in ("LEFT_KNEE_RAISED", "RIGHT_KNEE_RAISED")):
                    if any(v.get('zone_type') in ('monitored', 'restricted') for v in violations):
                        behavior_label = "AGGRESSIVE_POSE"
                        behavior_score = max(behavior_score, AGGRESSIVE_SCORE_THRESHOLD)

            # Process violations with behavior context
            if violations:
                for violation in violations:
                    severity = violation['severity']
                    zone_type = violation.get('zone_type', 'restricted')

                    # Escalate severity based on behavior (if enabled)
                    if ENABLE_BEHAVIOR_ANALYSIS:
                        if behavior_label == "AGGRESSIVE_POSE":
                            if zone_type == 'restricted':
                                severity = "CRITICAL"
                                violation['type'] = "AGGRESSIVE_INTRUSION"
                            else:
                                severity = "HIGH"
                                violation['type'] = "AGGRESSIVE_BEHAVIOR"
                        elif behavior_label == "SUSPICIOUS_POSE":
                            if zone_type == 'restricted':
                                severity = "HIGH"
                                violation['type'] = "SUSPICIOUS_INTRUSION"
                            else:
                                severity = "MEDIUM"
                                violation['type'] = "SUSPICIOUS_BEHAVIOR"

                    violation['severity'] = severity

                    self.intrusion_count += 1

                    # Print real-time alert
                    print(f"\n🚨 INTRUSION ALERT!")
                    print(f"   Frame: {self.frame_count}")
                    print(f"   Zone: {violation['zone']} ({zone_type})")
                    print(f"   Type: {violation['type']}")
                    print(f"   Severity: {severity}")
                    if behaviors:
                        print(f"   Behaviors: {', '.join(behaviors)} (score={behavior_score:.2f})")
                    print(f"   Person ID: {person_id}")
                    print(f"   CPU: {cpu:.1f}% | Memory: {mem:.1f}%")
                    if severity in ['CRITICAL', 'HIGH']:
                        print(f"   💡 Press 'S' to save screenshot")

                    # Log alert
                    self.alerts.append({
                        'frame': self.frame_count,
                        'type': 'ZONE_INTRUSION',
                        'zone': violation['zone'],
                        'zone_type': zone_type,
                        'violation_type': violation['type'],
                        'severity': severity,
                        'person_id': person_id,
                        'behavior_score': behavior_score,
                        'behaviors': behaviors,
                        'cpu': cpu,
                        'memory': mem
                    })

                # Draw alert box
                color = (0, 0, 255) if severity in ['HIGH', 'CRITICAL'] else (0, 165, 255)
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

                # Label with violation type and behavior
                label_text = f"{violation['type']}"
                cv2.putText(annotated, label_text,
                            (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if behaviors:
                    behavior_text = f"{behavior_label} ({behavior_score:.2f})"
                    cv2.putText(annotated, behavior_text,
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Normal person - draw green box
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(annotated, "Person",
                            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Frame timing
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        avg_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0.0

        # Display info
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Intrusions: {self.intrusion_count}",
            f"Screenshots: {self.screenshot_count}",
            f"FPS: {avg_fps:.1f}"
        ]
        y_offset = 30
        for line in info_lines:
            cv2.putText(annotated, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        cv2.putText(annotated, "S=Screenshot | Q=Quit",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated

    # ---------- RUN / REPORT ----------

    def run(self, video_source=0):
        """Main detection loop"""
        print("=" * 70)
        print("ZONE-BASED INTRUSION + BEHAVIOR DETECTION SYSTEM (DUAL-MODE)")
        print("=" * 70 + "\n")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ Cannot open {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
        if fps > 0:
            print(f"⏱️ Duration: {total_frames / fps:.1f}s\n")
        print("▶️ Processing... (Q=Quit, S=Screenshot)\n")
        print("=" * 70 + "\n")

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            annotated = self.process_frame(frame)

            cv2.imshow("Intrusion Detection System", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(annotated, frame_num)

            if frame_num % 50 == 0:
                print(f"  ✓ Processed {frame_num}/{total_frames} frames...")

        cap.release()
        cv2.destroyAllWindows()

        if self.screenshot_count > 0:
            print(f"\n📸 Total screenshots saved: {self.screenshot_count}")

        self.print_report()

    def print_report(self):
        """Print final report and save JSON"""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("INTRUSION DETECTION REPORT")
        print("=" * 70 + "\n")

        print("📊 SECURITY ALERTS:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Zone intrusions detected: {self.intrusion_count}")
        print(f"  Total alerts: {len(self.alerts)}")
        print(f"  Screenshots saved: {self.screenshot_count}")

        if self.alerts:
            severity_counts = defaultdict(int)
            for alert in self.alerts:
                severity_counts[alert['severity']] += 1

            print("\n  Alerts by Severity:")
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                if severity in severity_counts:
                    print(f"    • {severity}: {severity_counts[severity]}")

            zone_counts = defaultdict(int)
            for alert in self.alerts:
                zone_counts[alert['zone']] += 1

            print("\n  Zone Intrusion Details:")
            for zone, count in zone_counts.items():
                print(f"    • {zone}: {count} violations")

            type_counts = defaultdict(int)
            for alert in self.alerts:
                type_counts[alert['violation_type']] += 1

            print("\n  Violation Types:")
            for vtype, count in type_counts.items():
                print(f"    • {vtype}: {count} events")

            behavior_counts = defaultdict(int)
            for alert in self.alerts:
                for behavior in alert['behaviors']:
                    behavior_counts[behavior] += 1

            if behavior_counts:
                print("\n  Detected Behaviors:")
                for behavior, count in sorted(behavior_counts.items()):
                    print(f"    • {behavior}: {count} occurrences")

        print("\n⚙️ PERFORMANCE:")
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Avg frame time: {avg_frame_time * 1000:.1f} ms")
        print(f"  Total runtime: {elapsed:.1f} s")

        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0.0
        peak_cpu = max(self.cpu_usage) if self.cpu_usage else 0.0
        avg_mem = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0.0
        peak_mem = max(self.memory_usage) if self.memory_usage else 0.0

        if self.cpu_usage:
            print(f"  Average CPU: {avg_cpu:.1f}%")
            print(f"  Peak CPU: {peak_cpu:.1f}%")
        if self.memory_usage:
            print(f"  Average Memory: {avg_mem:.1f}%")
            print(f"  Peak Memory: {peak_mem:.1f}%")

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'intrusion_count': self.intrusion_count,
            'total_alerts': len(self.alerts),
            'screenshots_saved': self.screenshot_count,
            'elapsed_seconds': elapsed,
            'zones': self.zones,
            'config': {
                'enable_behavior_analysis': ENABLE_BEHAVIOR_ANALYSIS,
                'detection_conf': DETECTION_CONF,
                'hand_raise_distance': HAND_RAISE_DISTANCE,
                'hand_raise_distance_ratio': HAND_RAISE_DISTANCE_RATIO,
                'crouch_distance': CROUCH_DISTANCE,
                'stance_width': STANCE_WIDTH,
                'extreme_bend_distance': EXTREME_BEND_DISTANCE,
                'knee_raise_height': KNEE_RAISE_HEIGHT,
                'knee_raise_height_ratio': KNEE_RAISE_HEIGHT_RATIO,
                'aggressive_score_threshold': AGGRESSIVE_SCORE_THRESHOLD,
                'warning_score_threshold': WARNING_SCORE_THRESHOLD,
                'behavior_memory_frames': BEHAVIOR_MEMORY_FRAMES,
            },
            'performance': {
                'avg_fps': avg_fps,
                'avg_frame_time_ms': avg_frame_time * 1000,
                'avg_cpu': avg_cpu,
                'peak_cpu': peak_cpu,
                'avg_memory': avg_mem,
                'peak_memory': peak_mem,
            },
            'alerts': self.alerts
        }

        with open('intrusion_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n✓ Report saved: intrusion_report.json")
        print("=" * 70)
        print("END OF REPORT")
        print("=" * 70 + "\n")


# ---------- INTERACTIVE ZONE CREATOR (OPTIONAL) ----------

def create_custom_zones(video_path):
    """Interactive tool to define zones on first frame of video"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Cannot read video for zone creation.")
        return None

    zones = []
    current_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            print(f"Point added: ({x}, {y})")

    cv2.namedWindow('Define Zones')
    cv2.setMouseCallback('Define Zones', mouse_callback)

    print("\n=== ZONE DEFINITION TOOL ===")
    print("Click to add zone corners")
    print("Press 'r' = save as RESTRICTED zone")
    print("Press 'm' = save as MONITORED zone")
    print("Press 'c' = clear current zone points")
    print("Press 'q' = finish and start detection\n")

    while True:
        display = frame.copy()

        # Draw existing zones
        for zone in zones:
            pts = np.array(zone['polygon'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, zone['color'], 2)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], zone['color'])
            display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)

        # Draw current points
        for i, pt in enumerate(current_points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, current_points[i - 1], pt, (0, 255, 0), 2)

        cv2.imshow('Define Zones', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if len(current_points) >= 3:
                zones.append({
                    'name': f'Restricted Zone {len(zones) + 1}',
                    'polygon': current_points.copy(),
                    'type': 'restricted',
                    'color': (0, 0, 255)
                })
                print(f"✓ Restricted zone added with {len(current_points)} points")
                current_points = []
            else:
                print("Need at least 3 points to form a zone.")
        elif key == ord('m'):
            if len(current_points) >= 3:
                zones.append({
                    'name': f'Monitored Zone {len(zones) + 1}',
                    'polygon': current_points.copy(),
                    'type': 'monitored',
                    'color': (0, 165, 255)
                })
                print(f"✓ Monitored zone added with {len(current_points)} points")
                current_points = []
            else:
                print("Need at least 3 points to form a zone.")
        elif key == ord('c'):
            current_points = []
            print("Current points cleared")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return zones if zones else None


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_source = sys.argv[1]

        print("\nDo you want to define custom zones? (y/n): ", end="")
        try:
            response = input().strip().lower()
        except EOFError:
            response = 'n'

        if response == 'y':
            custom_zones = create_custom_zones(video_source)
            if custom_zones:
                detector = ZoneBehaviorDetector(zones=custom_zones)
            else:
                print("No custom zones defined. Using default zones.")
                detector = ZoneBehaviorDetector()
        else:
            detector = ZoneBehaviorDetector()

        detector.run(video_source=video_source)
    else:
        print("\n" + "=" * 70)
        print("ZONE-BASED INTRUSION DETECTION + BEHAVIOR ANALYSIS (DUAL-MODE)")
        print("=" * 70)
        print("\nUsage:")
        print("  python final_dualmode.py <video_path>")
        print("\nExample:")
        print("  python final_dualmode.py test_video.mp4")
        print("\nFeatures:")
        print("  • Zone-based intrusion detection")
        print("  • Pose-based behavior scoring (knee raise, crouch, wide stance, extreme bend)")
        print("  • Dual-mode thresholds (absolute + ratio-based) for hands and legs")
        print(f"  • Frame smoothing over {BEHAVIOR_MEMORY_FRAMES} frames")
        print("  • Screenshot capture: press 'S' during video")
        print("  • JSON report: intrusion_report.json")
        print("  • Optional interactive zone editor before running detection")
        print("\nControls during video:")
        print("  Q = Quit")
        print("  S = Save screenshot")
        print("\n" + "=" * 70 + "\n")
