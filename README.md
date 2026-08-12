# Football Analysis System ⚽

An AI/ML pipeline that analyzes football (soccer) match footage using computer vision to detect and track players, referees, and the ball, assign players to teams, estimate camera movement, transform pixel coordinates into real-world distances, and compute player speed, distance covered, and team ball possession.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple" />
  <img src="https://img.shields.io/badge/OpenCV-4.10-green" />
  <img src="https://img.shields.io/badge/status-in--progress-yellow" />
</p>

---

## 📹 Demo / Sample Data

Input and output videos (too large for GitHub) are available here:

**Google Drive:** [Input & Output Videos](https://drive.google.com/drive/folders/1T_Gt5L0BPUUUNnk8XDrAwXyJ6X5KtbGV?usp=sharing)

**Custom-trained detection dataset (Roboflow):** [Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)

---

## 🧠 Overview

This project processes raw broadcast football footage frame-by-frame and produces an annotated output video with:

- Player, referee, and ball detection & tracking (with persistent IDs across frames)
- Team classification based on jersey color
- Ball possession tracking per player and per team
- Camera movement compensation
- Real-world (meters) speed and distance calculation per player via perspective transform

---

## 🏗️ Pipeline Architecture

```
Input Video
    │
    ▼
┌─────────────────────┐
│  1. Object Detection  │  YOLOv8 (custom-trained on Roboflow dataset)
│     & Tracking         │  ByteTrack (via `supervision`)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  2. Team Assignment   │  K-Means clustering on jersey pixel colors
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  3. Camera Movement   │  Optical Flow (Lucas-Kanade) to estimate
│     Estimation         │  camera pan/movement between frames
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  4. Perspective        │  Homography transform: pixel coordinates
│     Transformation      │  → real-world pitch coordinates (meters)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  5. Ball Possession   │  Nearest-player-to-ball heuristic per frame
│     Assignment          │  → team ball control %
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  6. Speed & Distance   │  Distance covered (meters) + speed (km/h)
│     Estimation           │  computed over a rolling frame window
└─────────────────────┘
    │
    ▼
Annotated Output Video (.mp4)
```

---

## 🔑 Key Concepts

### 1. Object Detection & Tracking
- **YOLOv8** (`ultralytics`) detects 4 classes: `player`, `referee`, `goalkeeper`, `ball`.
- The custom model (`models/best.pt`) is fine-tuned on a football-specific dataset from Roboflow, since a stock COCO-pretrained model doesn't distinguish players/referees/ball well.
- **ByteTrack** (via the `supervision` library) assigns a persistent `track_id` to each detected object across frames, enabling per-player trajectory analysis.
- Goalkeepers are remapped to the `player` class internally so team-assignment and possession logic treat them uniformly.
- Detection runs in batches (default batch size 20) for efficiency.

### 2. Team Assignment (K-Means Clustering)
- For each player, the top half of their bounding box (roughly the torso/jersey area) is cropped.
- **K-Means clustering (k=2)** is applied to the cropped pixel colors to separate the dominant jersey color from the background.
- Each player is then assigned to **Team 1** or **Team 2** based on which cluster centroid their jersey color is closest to.
- Team colors are cached per `track_id` to keep classification stable and avoid re-computing every frame.

### 3. Camera Movement Estimation
- Since broadcast cameras pan/zoom to follow play, raw pixel positions alone don't reflect true player movement.
- **Optical flow (Lucas-Kanade method via OpenCV)** tracks a set of static background feature points (e.g. pitch lines, advertising boards) between consecutive frames to estimate how much the camera itself moved.
- This camera movement is then subtracted from each object's raw position to get an **adjusted position** that reflects true on-pitch movement, not camera panning.
- Camera movement stubs are cached (`stubs/camera_movement_stub.pkl`) to avoid recomputation on repeated runs of the same video.

### 4. Perspective Transformation
- Broadcast footage is shot at an angle, so pixel distances don't correspond linearly to real-world distances (players farther from the camera appear smaller and closer together in pixel space).
- A **homography matrix** (`cv2.getPerspectiveTransform`) maps a manually-calibrated quadrilateral of pixel coordinates (representing a known real-world rectangular area of the pitch) to actual pitch dimensions in meters.
- ⚠️ **Note:** These pixel calibration points are specific to each camera angle/video. When switching to a new input video, `view_transformer.py`'s `pixel_vertices` must be recalibrated to that video's framing, or all downstream speed/distance metrics will be inaccurate.

### 5. Ball Possession Assignment
- For each frame, the player whose bounding box is closest to the ball (within a max distance threshold) is assigned possession for that frame.
- This is a **frame-based proximity heuristic**, not a pass-counting algorithm — it does not track actual touches, passes, or trajectories, just closest player per frame.
- Team ball control % = (frames where a player from that team held "possession") / (total frames with any assignment).

### 6. Speed & Distance Estimation
- Player positions (after camera-movement adjustment and perspective transformation) are compared over a rolling window of frames (default: every 5 frames).
- Distance covered between the window's start and end position gives **distance in meters**.
- Speed is derived as `distance / time_elapsed`, converted to **km/h**.
- Both metrics are drawn on-screen below each tracked player.

---

## 📂 Project Structure

```
football_analysis/
├── main.py                          # Entry point — orchestrates the full pipeline
├── models/
│   └── best.pt                      # Custom-trained YOLOv8 weights (not tracked in git — see below)
├── trackers/
│   └── tracker.py                   # YOLOv8 detection + ByteTrack tracking + annotation drawing
├── team_assigner/
│   └── team_assigner.py             # K-Means jersey color clustering & team classification
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow camera movement estimation
├── view_transformer/
│   └── view_transformer.py          # Pixel → real-world perspective transform
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Nearest-player-to-ball possession logic
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Speed (km/h) & distance (m) calculation
├── utils/
│   └── video_utils.py, bbox_utils.py, etc.
├── training/
│   └── football_training_yolo_v5.ipynb  # Notebook to fine-tune YOLOv8 on the Roboflow dataset
├── input_videos/                    # Place raw match clips here (gitignored)
├── output_videos/                   # Annotated outputs are saved here (gitignored)
├── stubs/                           # Cached detection/tracking/camera-movement pickles (gitignored)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/amar2512003/FIFA26.git
cd FIFA26
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install ultralytics supervision opencv-python numpy scikit-learn roboflow
```


### 3. Get the model weights and sample videos
Since `models/best.pt` and sample input/output videos are too large for GitHub, download them from Google Drive:

👉 **[Google Drive — Notebooks & Input & Output Videos + Weights](https://drive.google.com/drive/folders/1T_Gt5L0BPUUUNnk8XDrAwXyJ6X5KtbGV?usp=sharing)**

Place the downloaded `best.pt` into `models/`, and any sample clips into `input_videos/`.

### 4. (Optional) Train your own detector
The custom object detector is trained on the **[Football Players Detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)** on Roboflow. To train your own weights, open `training/football_training_yolo_v5.ipynb`, add your own Roboflow API key (use an environment variable — don't hardcode it), and run through the notebook.

```python
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")
```

---

## ▶️ Usage

1. Place your input video in `input_videos/`.
2. Update the video path in `main.py`:
```python
video_frames = read_video('input_videos/your_video.mp4')
```
3. **If using a new/different video, recalibrate `view_transformer.py`'s `pixel_vertices`** to that video's camera framing (see Key Concepts #4), and clear old cached stubs:
```bash
rm stubs/*.pkl
```
4. Run the pipeline:
```bash
python3 main.py
```
5. Find the annotated output in `output_videos/`.

---

## ⚠️ Known Limitations

- **Perspective calibration is per-video.** The pixel-to-meter homography is hardcoded per camera angle and must be manually recalibrated for each new video source.
- **Possession is a proximity heuristic**, not true pass/touch detection.
- **Stub caching is video-specific.** Stubs (`stubs/*.pkl`) must be cleared or renamed per video, otherwise stale cached data from a previous (different-length) video will cause `IndexError`s.
- **Memory usage** scales with video length/resolution since frames are held fully in memory; very long or high-resolution clips may require a machine with more RAM or a streaming refactor.

---



## 👤 Author

**Amar Sinha**
