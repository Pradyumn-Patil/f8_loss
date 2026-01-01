# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ball Control Detection System for Figure-8 Cone Drills - analyzes video-derived parquet data to detect when a player loses control of the ball during Figure-8 soccer training drills.

**Cone Arrangement (bird's eye view, all cones on same horizontal line):**
```
Camera View (looking down from above):

     GATE 2              GATE 1            START
    +---------+        +---------+           o
    |         |        |         |       (px:1593)
    o ------- o  ----  o ------- o  ----
   L         R        L         R
(174)     (329)    (906)     (1065)

<==== FORWARD (player runs left) =========================>
<========================= BACKWARD (player runs right) ==>

Gate passage: player runs BETWEEN the left (L) and right (R) cones
Y-coords: ~861-878px (all roughly same height - straight line)
```

**Important**: Cone positions come from **JSON annotations** (`cone_annotations.json`), NOT from parquet detection. This provides pre-labeled cone roles for reliable gate detection.

## Package Structure

The codebase is organized into **3 clean folders**:

```
f8_loss/
├── detection/                    # FOLDER 1: Loss of control calculation logic
│   ├── ball_control_detector.py  # Core detection engine with detect_loss()
│   ├── figure8_cone_detector.py  # Cone role identification & gate tracking
│   ├── data_structures.py        # Data models, enums, classes
│   ├── data_loader.py            # Parquet/JSON loading
│   ├── config.py                 # Configuration classes
│   ├── csv_exporter.py           # CSV export functionality
│   └── turning_zones.py          # Elliptical turning zones for turn detection
│
├── annotation/                   # FOLDER 2: Cone annotation & visualization
│   ├── cone_annotator.py         # Interactive GUI annotation tool
│   ├── drill_visualizer.py       # Debug visualization (optional)
│   └── annotate_cones.py         # Annotation utilities
│
├── video/                        # FOLDER 3: Video generation with loss events
│   ├── annotate_with_json_cones.py  # Primary video annotation (JSON cones)
│   └── annotate_videos.py           # Alternative annotation (parquet cones)
│
├── __init__.py                   # Root exports (backwards compatible)
├── run_detection.py              # Main starter script
├── main.py                       # CLI entry point
├── testing.py                    # Validation framework
└── tests/                        # Test suite
```

## Common Commands

```bash
# Run detection for a single player
python run_detection.py abdullah_nasib

# Run detection for all players
python run_detection.py --all

# List available players and their data status
python run_detection.py --list

# Run validation tests against ground truth
python run_detection.py --test

# Run tests from parent directory (required for package imports)
cd .. && PYTHONPATH="." pytest f8_loss/tests/ -v

# Run a specific test file
cd .. && PYTHONPATH="." pytest f8_loss/tests/test_detector.py -v

# Run tests with coverage
cd .. && PYTHONPATH="." pytest f8_loss/tests/ --cov=f8_loss
```

## Architecture

### Core Detection Pipeline

```
cone_annotations.json ──> load_cone_annotations() ──> Figure8Layout
                                                           |
                                                           v
ball.parquet ────────────────────────────────────> BallControlDetector
                                                           |
pose.parquet ──> extract_ankle_positions() ────────────────|
                                                           |
cone.parquet ──> Figure8ConeDetector.identify_cone_roles() |
                         |                                 |
                         +─────────────────────────────────+
                                                           |
                                                           v
                                                    DetectionResult
                                                           |
                                                           v
                                                     CSVExporter
```

### Key Module Locations

**Detection entry points:**
- `detection/ball_control_detector.py` - `detect_ball_control()` convenience function
- `detection/ball_control_detector.py` - `BallControlDetector.detect()` main orchestrator

**Detection logic location:** The `detect_loss()` method in `detection/ball_control_detector.py` (starts at line 533) contains all loss detection logic. Modify ONLY this method to implement detection algorithms.

**BallControlDetector** delegates to:
- `detection/figure8_cone_detector.py` - Cone role identification + gate passage tracking
- `detection/data_loader.py` - Data preprocessing

### Imports

```python
# Option 1: Direct from subpackage (recommended)
from f8_loss.detection import BallControlDetector, ControlState, AppConfig
from f8_loss.detection.data_loader import load_cone_annotations

# Option 2: Backwards compatible from root
from f8_loss import BallControlDetector, ControlState, AppConfig, load_cone_annotations

# Annotation tools
from f8_loss.annotation import ConeAnnotator, DrillVisualizer

# Video generation
from f8_loss.video.annotate_with_json_cones import annotate_video_with_json_cones

# Turning zones
from f8_loss.detection.turning_zones import create_turning_zones, TurningZoneSet
```

### Data Structures (`detection/data_structures.py`)

**Cone Structures:**
- `ConeAnnotation`: Single cone with role, pixel position (px, py)
- `Figure8Layout`: Complete drill layout with gate line definitions

**Detection Structures:**
- `FrameData`: Per-frame analysis (positions, velocities, control scores, drill phase)
- `LossEvent`: Loss-of-control event with start/end frames, severity
- `GatePassage`: Gate crossing record with direction and quality score
- `DetectionResult`: Complete output container

**State Enums:**
- `ControlState`: CONTROLLED, TRANSITION, LOST, RECOVERING, UNKNOWN
- `DrillPhase`: AT_START, APPROACHING_G1, PASSING_G1, BETWEEN_GATES, etc.
- `DrillDirection`: FORWARD, BACKWARD, STATIONARY

**Turning Zone Structures (`detection/turning_zones.py`):**
- `TurningZone`: Elliptical zone with point-in-zone detection via ellipse equation
- `TurningZoneConfig`: Configuration for zone sizes and camera perspective stretch factors
- `TurningZoneSet`: Container for START and GATE2 zones with convenience methods

### Configuration (`detection/config.py`)

Factory methods: `AppConfig.for_figure8()`, `.with_strict_detection()`, `.with_lenient_detection()`

Key detection thresholds in `DetectionConfig`:
- `control_radius`: 120.0 (normal control distance)
- `loss_distance_threshold`: 200.0 (distance indicating loss)
- `loss_velocity_spike`: 100.0 (velocity indicating sudden loss)
- `loss_duration_frames`: 5 (frames to confirm sustained loss)

## Data Flow

**Input Files:**

| File | Format | Purpose | Coordinates |
|------|--------|---------|-------------|
| `cone_annotations.json` | JSON | Static cone positions with pre-labeled roles | Pixel (px, py) |
| `*_football.parquet` | Parquet | Ball positions per frame | Pixel + Field |
| `*_pose.parquet` | Parquet | 26 keypoints/person/frame (only ankles used) | Pixel + Field |
| `*_cone.parquet` | Parquet | **OPTIONAL** - for visualization only | Pixel + Field |

**JSON Cone Annotation Format:**
```json
{
  "video": "player_name_f8.MOV",
  "cones": {
    "start": {"bbox": {...}, "px": 1593, "py": 878},
    "gate1_left": {"px": 906, "py": 869},
    "gate1_right": {"px": 1065, "py": 870},
    "gate2_left": {"px": 174, "py": 861},
    "gate2_right": {"px": 329, "py": 865}
  }
}
```

**Output CSVs:**
- `loss_events.csv`: Detected loss events with timestamps and context
- `frame_analysis.csv`: Per-frame metrics and states
- `gate_passages.csv`: Gate crossing records
- `cone_roles.csv`: Cone positions and assigned roles

## Key Design Decisions

- **JSON Cone Annotations**: Uses pre-labeled `cone_annotations.json` instead of auto-detecting cone roles from parquet
- **Pixel Space for Gates**: Gate passage detection uses pixel coordinates (matches JSON format)
- **Field Space for Control**: Ball-ankle distance scoring uses field coordinates
- **Ankles only**: Uses ankle keypoints for ball-foot distance (`left_ankle`, `right_ankle`)
- **3-Folder Structure**: Code organized into detection/, annotation/, video/
- **Backwards Compatible**: Root `__init__.py` re-exports key classes
- **Control scoring**: Weighted combination: 60% distance, 25% velocity, 15% stability
- **Gate detection**: Line segment intersection between consecutive ball positions and gate line
- **Visualization optional**: OpenCV import wrapped in try/except; detection works without it

## CRITICAL: Visualization and Detection Logic Consistency

**The visualization logic in `video/annotate_with_json_cones.py` MUST use the same thresholds and logic as the detection in `detection/ball_control_detector.py`.**

The video annotation serves as a **debug tool** - what you see in the annotated video (FRONT/BEHIND labels, colors) should match exactly what the detection algorithm calculates. If they differ, debugging becomes impossible.

**Synchronized thresholds:**

| Parameter | Detection File | Visualization File | Value |
|-----------|---------------|-------------------|-------|
| Ball-behind threshold | `ball_control_detector.py:_behind_threshold` | `annotate_with_json_cones.py:BALL_POSITION_THRESHOLD` | 20.0px |
| Movement threshold | `ball_control_detector.py:_movement_threshold` | `annotate_with_json_cones.py:MOVEMENT_THRESHOLD` | 3.0px |

**When modifying detection logic:**
1. Update the threshold/logic in `detection/ball_control_detector.py`
2. Update the SAME threshold/logic in `video/annotate_with_json_cones.py`
3. Regenerate annotated video to verify visually

## Video Generation Requirements

**IMPORTANT**: When generating annotated videos, always use H.264 codec for compatibility with VS Code and modern video players.

**OpenCV creates incompatible mp4v format** - must convert using ffmpeg:

```python
# OpenCV writes mp4v (incompatible with many players)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# After writing, convert to H.264 with ffmpeg:
ffmpeg -y -i input.mp4 -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart output.mp4
```

**Required ffmpeg parameters:**
- `-c:v libx264`: H.264 video codec (universal compatibility)
- `-preset medium`: Balance between speed and compression
- `-crf 23`: Good quality (18-28 range, lower = better)
- `-pix_fmt yuv420p`: Standard pixel format for compatibility
- `-movflags +faststart`: Move moov atom for web streaming

**Video annotation scripts:**
- `video/annotate_videos.py`: Uses parquet cone detection (per-frame positions)
- `video/annotate_with_json_cones.py`: Uses JSON cone annotations (static positions) - automatically converts to H.264

## File Naming Convention

Player data follows pattern:
```
video_detection_pose_ball_cones/
  {player_name}_f8/
    {player_name}_f8_football.parquet
    {player_name}_f8_pose.parquet
    {player_name}_f8_cone.parquet
    cone_annotations.json
```

Some players use `{player_name}` without `_f8` suffix - `run_detection.py` handles both conventions.
