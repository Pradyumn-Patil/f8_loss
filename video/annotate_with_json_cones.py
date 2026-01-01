#!/usr/bin/env python3
"""
Video Annotation with JSON Cone Annotations for F8 Drill Analysis.

Creates annotated videos using:
- STATIC cone positions from JSON annotations (same position every frame)
- DYNAMIC ball positions from parquet detection
- DYNAMIC pose skeleton from parquet detection
- LEFT SIDEBAR showing all object coordinates in real-time

This provides a stable drill layout reference while showing player/ball movement.

Usage:
    python annotate_with_json_cones.py abdullah_nasib_f8
    python annotate_with_json_cones.py --list
    python annotate_with_json_cones.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from dataclasses import dataclass
from collections import deque
import math
from tqdm import tqdm

# Import turning zones from detection module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.turning_zones import (
    TurningZoneSet, TurningZoneConfig, create_turning_zones,
    draw_turning_zones, START_ZONE_COLOR, GATE2_ZONE_COLOR, ZONE_HIGHLIGHT_COLOR
)
from video.drill_event_tracker import (
    DrillEventTracker, DrillEvent, DrillEventType,
    draw_debug_axes, draw_event_log, draw_cone_threshold_lines
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnnotationConfig:
    """Configuration for annotation styles."""
    # Sidebar settings
    SIDEBAR_WIDTH: int = 300
    SIDEBAR_BG_COLOR: Tuple[int, int, int] = (25, 25, 25)  # Dark gray
    SIDEBAR_HEADER_COLOR: Tuple[int, int, int] = (80, 80, 80)  # Lighter gray
    SIDEBAR_LINE_HEIGHT: int = 24
    SIDEBAR_FONT_SCALE: float = 0.55
    SIDEBAR_PADDING: int = 12

    # Colors (BGR format for OpenCV)
    BALL_COLOR: Tuple[int, int, int] = (0, 255, 0)           # Green
    START_CONE_COLOR: Tuple[int, int, int] = (0, 255, 255)   # Yellow
    GATE1_COLOR: Tuple[int, int, int] = (255, 150, 0)        # Blue
    GATE2_COLOR: Tuple[int, int, int] = (255, 0, 255)        # Magenta
    POSE_KEYPOINT_COLOR: Tuple[int, int, int] = (255, 0, 255)
    POSE_SKELETON_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)       # White
    TEXT_BG_COLOR: Tuple[int, int, int] = (0, 0, 0)          # Black

    # Turning zone colors
    START_ZONE_COLOR: Tuple[int, int, int] = (200, 200, 0)     # Teal/Cyan
    GATE2_ZONE_COLOR: Tuple[int, int, int] = (200, 100, 200)   # Purple
    ZONE_HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 255, 255) # Yellow (ball inside)
    ZONE_ALPHA: float = 0.25                                    # Zone transparency

    # Sizes
    BBOX_THICKNESS: int = 2
    SKELETON_THICKNESS: int = 2
    KEYPOINT_RADIUS: int = 4
    CONE_RADIUS: int = 12
    GATE_LINE_THICKNESS: int = 3
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 1

    # Confidence thresholds
    MIN_KEYPOINT_CONFIDENCE: float = 0.3
    MIN_BBOX_CONFIDENCE: float = 0.1

    # Momentum arrow settings
    DRAW_MOMENTUM_ARROW: bool = True
    MOMENTUM_THICKNESS: int = 8  # Thick arrow for visibility
    MOMENTUM_LOOKBACK_FRAMES: int = 10
    MOMENTUM_SCALE: float = 3.0  # Scale factor for visibility
    MOMENTUM_MAX_LENGTH: int = 150  # Max arrow length in pixels
    MOMENTUM_MIN_LENGTH: int = 5  # Min movement to show arrow (avoid jitter)

    # Momentum color gradient settings (green=slow, yellow=medium, red=fast)
    MOMENTUM_COLOR_LOW: Tuple[int, int, int] = (0, 255, 0)    # Green (BGR) - slow
    MOMENTUM_COLOR_MID: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR) - medium
    MOMENTUM_COLOR_HIGH: Tuple[int, int, int] = (0, 0, 255)   # Red (BGR) - fast
    MOMENTUM_SPEED_LOW: float = 5.0    # Below this = green (px/lookback)
    MOMENTUM_SPEED_HIGH: float = 80.0  # Above this = red (px/lookback)

    # Ball position relative to player settings
    DRAW_BALL_POSITION: bool = True
    BALL_POSITION_FRONT_COLOR: Tuple[int, int, int] = (0, 255, 0)    # Green (BGR)
    BALL_POSITION_BEHIND_COLOR: Tuple[int, int, int] = (0, 0, 255)   # Red (BGR)
    BALL_POSITION_ALIGNED_COLOR: Tuple[int, int, int] = (0, 255, 255) # Yellow (BGR)
    BALL_POSITION_NEUTRAL_COLOR: Tuple[int, int, int] = (180, 180, 180) # Gray (stationary)
    BALL_HIP_LINE_THICKNESS: int = 2
    BALL_POSITION_THRESHOLD: float = 20.0  # Pixels for "aligned" detection
    MOVEMENT_THRESHOLD: float = 3.0  # Min movement to determine direction
    DIVIDER_LINE_HEIGHT: int = 100  # Height of vertical dividing line through hip

    # Ball-behind duration counter settings (debug feature)
    BEHIND_COUNTER_PERSIST_SECONDS: float = 3.0  # Show count for 3 sec after behind ends
    BEHIND_COUNTER_FONT_SCALE: float = 1.2
    BEHIND_COUNTER_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR) when active
    BEHIND_COUNTER_PERSIST_COLOR: Tuple[int, int, int] = (0, 200, 255)  # Yellow when persisting
    BEHIND_COUNTER_POS_X: int = 50  # Position on video (relative to sidebar)
    BEHIND_COUNTER_POS_Y: int = 100

    # Edge zone visualization settings (matching detection thresholds)
    EDGE_MARGIN: int = 50                    # Pixels from edge for "in zone" (matches detection)
    EDGE_ZONE_COLOR: Tuple[int, int, int] = (0, 0, 255)      # Red for danger zone (BGR)
    EDGE_ZONE_ALPHA: float = 0.15            # Subtle transparency for zone overlay
    EDGE_COUNTER_FONT_SCALE: float = 1.2
    EDGE_COUNTER_COLOR: Tuple[int, int, int] = (0, 0, 255)   # Red when active
    EDGE_COUNTER_PERSIST_COLOR: Tuple[int, int, int] = (0, 200, 255)  # Yellow when persisting
    EDGE_COUNTER_PERSIST_SECONDS: float = 3.0
    EDGE_COUNTER_POS_X: int = 50
    EDGE_COUNTER_POS_Y: int = 150            # Below BEHIND counter (which is at Y=100)

    # Drill event tracking settings
    DRAW_DEBUG_AXES: bool = True
    DEBUG_AXES_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    DEBUG_AXES_THICKNESS: int = 1
    DRAW_CONE_REFERENCE: bool = True         # Draw reference line at cone X position
    CONE_REFERENCE_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
    DRAW_EVENT_LOG: bool = True
    EVENT_LOG_MAX_EVENTS: int = 8
    # Threshold lines for pass above/below debug
    DRAW_THRESHOLD_LINES: bool = True        # Draw static X and Y threshold lines
    THRESHOLD_X_COLOR: Tuple[int, int, int] = (255, 0, 255)  # Magenta for X (crossing)
    THRESHOLD_Y_COLOR: Tuple[int, int, int] = (0, 255, 0)    # Green for Y (above/below)
    THRESHOLD_LINE_THICKNESS: int = 1


# Skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('nose', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    ('neck', 'hip'), ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'), ('right_ankle', 'right_heel'),
    ('left_ankle', 'left_big_toe'), ('right_ankle', 'right_big_toe'),
    ('left_big_toe', 'left_small_toe'), ('right_big_toe', 'right_small_toe'),
]

KEYPOINT_COLORS = {
    'head': (255, 200, 200),
    'torso': (200, 255, 200),
    'arms': (200, 200, 255),
    'legs': (255, 255, 200),
    'feet': (255, 200, 255),
}

KEYPOINT_BODY_PART = {
    'nose': 'head', 'left_eye': 'head', 'right_eye': 'head',
    'left_ear': 'head', 'right_ear': 'head', 'head': 'head',
    'neck': 'torso', 'left_shoulder': 'torso', 'right_shoulder': 'torso',
    'hip': 'torso', 'left_hip': 'torso', 'right_hip': 'torso',
    'left_elbow': 'arms', 'right_elbow': 'arms',
    'left_wrist': 'arms', 'right_wrist': 'arms',
    'left_knee': 'legs', 'right_knee': 'legs',
    'left_ankle': 'legs', 'right_ankle': 'legs',
    'left_big_toe': 'feet', 'right_big_toe': 'feet',
    'left_small_toe': 'feet', 'right_small_toe': 'feet',
    'left_heel': 'feet', 'right_heel': 'feet',
}

# Keypoints to track in sidebar (important for ball control analysis)
TRACKED_KEYPOINTS = [
    ('left_ankle', 'L_ANKLE'),
    ('right_ankle', 'R_ANKLE'),
    ('left_big_toe', 'L_TOE'),
    ('right_big_toe', 'R_TOE'),
    ('nose', 'HEAD'),
]


# ============================================================================
# JSON Cone Loading
# ============================================================================

@dataclass
class ConeAnnotation:
    """Single cone from JSON annotation."""
    role: str
    px: float
    py: float
    bbox: Optional[Dict[str, int]] = None  # Bounding box with x1, y1, x2, y2

    @property
    def y2(self) -> float:
        """Bottom edge of cone bounding box (or py if no bbox)."""
        if self.bbox and 'y2' in self.bbox:
            return self.bbox['y2']
        return self.py


@dataclass
class Figure8Layout:
    """Complete Figure-8 layout from JSON."""
    start: ConeAnnotation
    gate1_left: ConeAnnotation
    gate1_right: ConeAnnotation
    gate2_left: ConeAnnotation
    gate2_right: ConeAnnotation

    @classmethod
    def from_json(cls, json_path: Path) -> 'Figure8Layout':
        """Load from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        cones = data['cones']
        return cls(
            start=ConeAnnotation('start', cones['start']['px'], cones['start']['py'], cones['start'].get('bbox')),
            gate1_left=ConeAnnotation('gate1_left', cones['gate1_left']['px'], cones['gate1_left']['py'], cones['gate1_left'].get('bbox')),
            gate1_right=ConeAnnotation('gate1_right', cones['gate1_right']['px'], cones['gate1_right']['py'], cones['gate1_right'].get('bbox')),
            gate2_left=ConeAnnotation('gate2_left', cones['gate2_left']['px'], cones['gate2_left']['py'], cones['gate2_left'].get('bbox')),
            gate2_right=ConeAnnotation('gate2_right', cones['gate2_right']['px'], cones['gate2_right']['py'], cones['gate2_right'].get('bbox')),
        )

    @property
    def gate1_width(self) -> float:
        return np.sqrt((self.gate1_right.px - self.gate1_left.px)**2 +
                       (self.gate1_right.py - self.gate1_left.py)**2)

    @property
    def gate2_width(self) -> float:
        return np.sqrt((self.gate2_right.px - self.gate2_left.px)**2 +
                       (self.gate2_right.py - self.gate2_left.py)**2)

    @property
    def start_cone(self) -> ConeAnnotation:
        """Alias for start (for compatibility with detection module)."""
        return self.start

    @property
    def gate2_center(self) -> Tuple[float, float]:
        """Center point of Gate 2 in pixel coords."""
        return (
            (self.gate2_left.px + self.gate2_right.px) / 2,
            (self.gate2_left.py + self.gate2_right.py) / 2
        )


# ============================================================================
# Data Loaders
# ============================================================================

def load_ball_data(parquet_path: Path) -> pd.DataFrame:
    """Load ball detection data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_id', 'x1', 'y1', 'x2', 'y2', 'confidence']].copy()


def load_pose_data(parquet_path: Path) -> pd.DataFrame:
    """Load pose keypoint data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_idx', 'person_id', 'keypoint_name', 'x', 'y', 'confidence']].copy()


def prepare_pose_lookup(pose_df: pd.DataFrame) -> Dict[int, Dict[int, Dict[str, Tuple[float, float, float]]]]:
    """Create efficient lookup for pose data."""
    lookup = {}
    for _, row in pose_df.iterrows():
        frame_id = int(row['frame_idx'])
        person_id = int(row['person_id'])
        keypoint = row['keypoint_name']

        if frame_id not in lookup:
            lookup[frame_id] = {}
        if person_id not in lookup[frame_id]:
            lookup[frame_id][person_id] = {}

        lookup[frame_id][person_id][keypoint] = (
            float(row['x']),
            float(row['y']),
            float(row['confidence'])
        )
    return lookup


# ============================================================================
# Sidebar Drawing Functions
# ============================================================================

def draw_sidebar_section_header(frame: np.ndarray, y: int, text: str,
                                color: Tuple[int, int, int], config: AnnotationConfig) -> int:
    """Draw a section header in the sidebar. Returns next y position."""
    # Draw separator line
    cv2.line(frame, (config.SIDEBAR_PADDING, y),
             (config.SIDEBAR_WIDTH - config.SIDEBAR_PADDING, y),
             config.SIDEBAR_HEADER_COLOR, 1)

    y += 18
    cv2.putText(frame, text, (config.SIDEBAR_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                color, 1, cv2.LINE_AA)

    return y + 8


def draw_sidebar_row(frame: np.ndarray, y: int, label: str,
                     x_coord: Optional[float], y_coord: Optional[float],
                     color: Tuple[int, int, int], config: AnnotationConfig) -> int:
    """Draw a coordinate row in the sidebar. Returns next y position."""
    # Format: "LABEL:   (xxxx, yyyy)" or "LABEL:   --"
    if x_coord is not None and y_coord is not None:
        coord_str = f"({int(x_coord):4d}, {int(y_coord):4d})"
    else:
        coord_str = "   --"

    # Draw label (left aligned)
    label_x = config.SIDEBAR_PADDING + 5
    cv2.putText(frame, f"{label}:", (label_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.05,
                color, 1, cv2.LINE_AA)

    # Draw coordinates (right side)
    coord_x = config.SIDEBAR_WIDTH - 130
    cv2.putText(frame, coord_str, (coord_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.05,
                (200, 200, 200), 1, cv2.LINE_AA)

    return y + config.SIDEBAR_LINE_HEIGHT


def draw_sidebar(frame: np.ndarray, frame_id: int, layout: Figure8Layout,
                 ball_center: Optional[Tuple[float, float]],
                 pose_keypoints: Dict[str, Tuple[float, float, float]],
                 config: AnnotationConfig,
                 active_zone: Optional[str] = None,
                 ball_position_result: Optional['BallPositionResult'] = None,
                 drill_events: Optional[List['DrillEvent']] = None) -> None:
    """
    Draw the sidebar with all object coordinates.

    Args:
        frame: The full canvas (sidebar + video)
        frame_id: Current frame number
        layout: Figure8 cone layout
        ball_center: (x, y) of ball center or None if not detected
        pose_keypoints: Dict of keypoint_name -> (x, y, confidence)
        config: Annotation configuration
        active_zone: Name of zone ball is in ("START", "GATE2", or None)
        ball_position_result: Result from ball position detection (front/behind)
        drill_events: List of recent drill events (cone crossings, turns)
    """
    # Fill sidebar background
    frame[:, :config.SIDEBAR_WIDTH] = config.SIDEBAR_BG_COLOR

    y = 25

    # Frame number header
    cv2.putText(frame, f"FRAME: {frame_id}", (config.SIDEBAR_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 35

    # ===== CONES SECTION =====
    y = draw_sidebar_section_header(frame, y, "CONES (static)",
                                    config.START_CONE_COLOR, config)
    y += 5

    # Start cone
    y = draw_sidebar_row(frame, y, "START", layout.start.px, layout.start.py,
                         config.START_CONE_COLOR, config)

    # Gate 1 cones
    y = draw_sidebar_row(frame, y, "G1-L", layout.gate1_left.px, layout.gate1_left.py,
                         config.GATE1_COLOR, config)
    y = draw_sidebar_row(frame, y, "G1-R", layout.gate1_right.px, layout.gate1_right.py,
                         config.GATE1_COLOR, config)

    # Gate 2 cones
    y = draw_sidebar_row(frame, y, "G2-L", layout.gate2_left.px, layout.gate2_left.py,
                         config.GATE2_COLOR, config)
    y = draw_sidebar_row(frame, y, "G2-R", layout.gate2_right.px, layout.gate2_right.py,
                         config.GATE2_COLOR, config)

    y += 10

    # ===== BALL SECTION =====
    y = draw_sidebar_section_header(frame, y, "BALL (dynamic)",
                                    config.BALL_COLOR, config)
    y += 5

    ball_x = ball_center[0] if ball_center else None
    ball_y = ball_center[1] if ball_center else None
    y = draw_sidebar_row(frame, y, "BALL", ball_x, ball_y,
                         config.BALL_COLOR, config)

    y += 10

    # ===== POSE SECTION =====
    y = draw_sidebar_section_header(frame, y, "POSE (dynamic)",
                                    config.POSE_SKELETON_COLOR, config)
    y += 5

    for kp_name, display_name in TRACKED_KEYPOINTS:
        if kp_name in pose_keypoints:
            x, ycoord, conf = pose_keypoints[kp_name]
            if conf >= config.MIN_KEYPOINT_CONFIDENCE:
                body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
                color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
                y = draw_sidebar_row(frame, y, display_name, x, ycoord, color, config)
            else:
                y = draw_sidebar_row(frame, y, display_name, None, None,
                                     (100, 100, 100), config)
        else:
            y = draw_sidebar_row(frame, y, display_name, None, None,
                                 (100, 100, 100), config)

    y += 15

    # ===== GATE INFO SECTION =====
    y = draw_sidebar_section_header(frame, y, "GATE INFO",
                                    (150, 150, 150), config)
    y += 5

    cv2.putText(frame, f"G1 width: {layout.gate1_width:.0f}px",
                (config.SIDEBAR_PADDING + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1,
                config.GATE1_COLOR, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    cv2.putText(frame, f"G2 width: {layout.gate2_width:.0f}px",
                (config.SIDEBAR_PADDING + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1,
                config.GATE2_COLOR, 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + 10

    # ===== TURNING ZONE SECTION =====
    y = draw_sidebar_section_header(frame, y, "TURNING ZONE",
                                    (150, 150, 150), config)
    y += 5

    # Show which zone ball is in
    if active_zone == "START":
        zone_text = "Ball in: START"
        zone_color = config.START_ZONE_COLOR
    elif active_zone == "GATE2":
        zone_text = "Ball in: GATE2"
        zone_color = config.GATE2_ZONE_COLOR
    else:
        zone_text = "Ball in: --"
        zone_color = (100, 100, 100)

    cv2.putText(frame, zone_text,
                (config.SIDEBAR_PADDING + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                zone_color, 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + 10

    # ===== BALL POSITION SECTION (Front/Behind) =====
    y = draw_sidebar_section_header(frame, y, "BALL POSITION",
                                    (150, 150, 150), config)
    y += 5

    if ball_position_result is not None:
        # Position status (FRONT/BEHIND/ALIGNED)
        pos_text = f"Status: {ball_position_result.position}"
        cv2.putText(frame, pos_text,
                    (config.SIDEBAR_PADDING + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    ball_position_result.color, 1, cv2.LINE_AA)
        y += config.SIDEBAR_LINE_HEIGHT

        # Movement direction
        dir_text = f"Moving: {ball_position_result.movement_direction or 'STATIONARY'}"
        dir_color = ball_position_result.color if ball_position_result.movement_direction else (100, 100, 100)
        cv2.putText(frame, dir_text,
                    (config.SIDEBAR_PADDING + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1,
                    dir_color, 1, cv2.LINE_AA)
        y += config.SIDEBAR_LINE_HEIGHT

        # Delta X (ball-hip horizontal distance)
        delta_text = f"Delta X: {ball_position_result.ball_hip_delta_x:+.0f}px"
        cv2.putText(frame, delta_text,
                    (config.SIDEBAR_PADDING + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1,
                    (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Status: --",
                    (config.SIDEBAR_PADDING + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    (100, 100, 100), 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + 10

    # ===== DRILL EVENTS SECTION =====
    if config.DRAW_EVENT_LOG and drill_events is not None:
        y = draw_event_log(
            frame,
            events=drill_events,
            start_y=y,
            sidebar_width=config.SIDEBAR_WIDTH,
            sidebar_padding=config.SIDEBAR_PADDING,
            line_height=config.SIDEBAR_LINE_HEIGHT,
            font_scale=config.SIDEBAR_FONT_SCALE,
            header_color=config.SIDEBAR_HEADER_COLOR,
            max_events=config.EVENT_LOG_MAX_EVENTS
        )

    # Draw vertical separator line between sidebar and video
    cv2.line(frame, (config.SIDEBAR_WIDTH - 1, 0),
             (config.SIDEBAR_WIDTH - 1, frame.shape[0]),
             (60, 60, 60), 2)


# ============================================================================
# Video Drawing Functions
# ============================================================================

def draw_bbox(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              color: Tuple[int, int, int], label: str, config: AnnotationConfig,
              x_offset: int = 0) -> None:
    """Draw a bounding box with label."""
    x1, y1, x2, y2 = int(x1) + x_offset, int(y1), int(x2) + x_offset, int(y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    cv2.rectangle(frame, (x1, y1 - text_height - 8),
                  (x1 + text_width + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.TEXT_BG_COLOR, config.FONT_THICKNESS)


def draw_cone_marker(frame: np.ndarray, cone: ConeAnnotation, label: str,
                     color: Tuple[int, int, int], config: AnnotationConfig,
                     x_offset: int = 0) -> None:
    """Draw a cone marker as bounding box (no label)."""
    if cone.bbox:
        # Use actual bounding box from JSON annotation
        x1 = int(cone.bbox['x1']) + x_offset
        y1 = int(cone.bbox['y1'])
        x2 = int(cone.bbox['x2']) + x_offset
        y2 = int(cone.bbox['y2'])
    else:
        # Fallback: create square from center point
        half_size = config.CONE_RADIUS
        x1 = int(cone.px) + x_offset - half_size
        y1 = int(cone.py) - half_size
        x2 = int(cone.px) + x_offset + half_size
        y2 = int(cone.py) + half_size

    # Draw rectangle outline (not filled) so cone is visible underneath
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_gate_line(frame: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float],
                   color: Tuple[int, int, int], label: str, config: AnnotationConfig,
                   x_offset: int = 0) -> None:
    """Draw a gate line between two cones."""
    pt1 = (int(p1[0]) + x_offset, int(p1[1]))
    pt2 = (int(p2[0]) + x_offset, int(p2[1]))

    # Draw dashed-style gate line
    cv2.line(frame, pt1, pt2, color, config.GATE_LINE_THICKNESS)

    # Draw label in center of gate
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2 - 25  # Above the line

    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )

    # Background
    cv2.rectangle(frame,
                  (center_x - text_width // 2 - 3, center_y - text_height - 3),
                  (center_x + text_width // 2 + 3, center_y + 3),
                  (0, 0, 0), -1)
    cv2.putText(frame, label,
                (center_x - text_width // 2, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_json_cones(frame: np.ndarray, layout: Figure8Layout, config: AnnotationConfig,
                    x_offset: int = 0) -> None:
    """Draw all JSON-annotated cones with gate lines."""
    # Draw Start cone
    draw_cone_marker(frame, layout.start, "START", config.START_CONE_COLOR, config, x_offset)

    # Draw Gate 1 cones and line
    draw_cone_marker(frame, layout.gate1_left, "G1-L", config.GATE1_COLOR, config, x_offset)
    draw_cone_marker(frame, layout.gate1_right, "G1-R", config.GATE1_COLOR, config, x_offset)
    draw_gate_line(frame,
                   (layout.gate1_left.px, layout.gate1_left.py),
                   (layout.gate1_right.px, layout.gate1_right.py),
                   config.GATE1_COLOR, "GATE 1", config, x_offset)

    # Draw Gate 2 cones and line
    draw_cone_marker(frame, layout.gate2_left, "G2-L", config.GATE2_COLOR, config, x_offset)
    draw_cone_marker(frame, layout.gate2_right, "G2-R", config.GATE2_COLOR, config, x_offset)
    draw_gate_line(frame,
                   (layout.gate2_left.px, layout.gate2_left.py),
                   (layout.gate2_right.px, layout.gate2_right.py),
                   config.GATE2_COLOR, "GATE 2", config, x_offset)


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[float, float, float]],
                  config: AnnotationConfig, x_offset: int = 0) -> None:
    """Draw pose skeleton with keypoints (coordinates shown in sidebar)."""
    # Draw connections first
    for kp1_name, kp2_name in SKELETON_CONNECTIONS:
        if kp1_name in keypoints and kp2_name in keypoints:
            x1, y1, conf1 = keypoints[kp1_name]
            x2, y2, conf2 = keypoints[kp2_name]

            if conf1 >= config.MIN_KEYPOINT_CONFIDENCE and conf2 >= config.MIN_KEYPOINT_CONFIDENCE:
                pt1 = (int(x1) + x_offset, int(y1))
                pt2 = (int(x2) + x_offset, int(y2))
                cv2.line(frame, pt1, pt2, config.POSE_SKELETON_COLOR, config.SKELETON_THICKNESS)

    # Draw keypoints (no coordinate labels - shown in sidebar)
    for kp_name, (x, y, conf) in keypoints.items():
        if conf >= config.MIN_KEYPOINT_CONFIDENCE:
            pt = (int(x) + x_offset, int(y))
            body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
            color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, color, -1)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, (0, 0, 0), 1)


def get_momentum_color(
    magnitude: float,
    config: AnnotationConfig
) -> Tuple[int, int, int]:
    """
    Calculate momentum arrow color based on speed magnitude.
    Uses smooth gradient: green (slow) -> yellow (medium) -> red (fast).

    Args:
        magnitude: Movement magnitude in pixels
        config: Annotation configuration with speed thresholds

    Returns:
        BGR color tuple
    """
    # Normalize magnitude to 0-1 range
    if magnitude <= config.MOMENTUM_SPEED_LOW:
        t = 0.0
    elif magnitude >= config.MOMENTUM_SPEED_HIGH:
        t = 1.0
    else:
        t = (magnitude - config.MOMENTUM_SPEED_LOW) / (config.MOMENTUM_SPEED_HIGH - config.MOMENTUM_SPEED_LOW)

    # Two-stage interpolation: green -> yellow -> red
    if t <= 0.5:
        # Green to Yellow (t: 0 -> 0.5)
        ratio = t * 2  # 0 to 1
        b = int(config.MOMENTUM_COLOR_LOW[0] + ratio * (config.MOMENTUM_COLOR_MID[0] - config.MOMENTUM_COLOR_LOW[0]))
        g = int(config.MOMENTUM_COLOR_LOW[1] + ratio * (config.MOMENTUM_COLOR_MID[1] - config.MOMENTUM_COLOR_LOW[1]))
        r = int(config.MOMENTUM_COLOR_LOW[2] + ratio * (config.MOMENTUM_COLOR_MID[2] - config.MOMENTUM_COLOR_LOW[2]))
    else:
        # Yellow to Red (t: 0.5 -> 1)
        ratio = (t - 0.5) * 2  # 0 to 1
        b = int(config.MOMENTUM_COLOR_MID[0] + ratio * (config.MOMENTUM_COLOR_HIGH[0] - config.MOMENTUM_COLOR_MID[0]))
        g = int(config.MOMENTUM_COLOR_MID[1] + ratio * (config.MOMENTUM_COLOR_HIGH[1] - config.MOMENTUM_COLOR_MID[1]))
        r = int(config.MOMENTUM_COLOR_MID[2] + ratio * (config.MOMENTUM_COLOR_HIGH[2] - config.MOMENTUM_COLOR_MID[2]))

    return (b, g, r)


def draw_momentum_arrow(
    frame: np.ndarray,
    current_hip: Tuple[float, float],
    previous_hip: Tuple[float, float],
    config: AnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw thick horizontal momentum arrow with color gradient.

    Arrow is always parallel to ground (horizontal only).
    Color changes based on speed: green (slow) -> yellow (medium) -> red (fast).

    Args:
        frame: Video frame to draw on
        current_hip: Current (x, y) position of hip
        previous_hip: Previous (x, y) position of hip (from N frames ago)
        config: Annotation configuration
        x_offset: Horizontal offset for sidebar
    """
    # Calculate horizontal displacement only (parallel to ground)
    dx = current_hip[0] - previous_hip[0]

    # Use absolute horizontal magnitude for speed calculation
    horizontal_magnitude = abs(dx)

    # Skip if horizontal movement too small (reduces jitter)
    if horizontal_magnitude < config.MOMENTUM_MIN_LENGTH:
        return

    # Get color based on horizontal speed
    color = get_momentum_color(horizontal_magnitude, config)

    # Scale and cap the arrow length
    scaled_length = min(horizontal_magnitude * config.MOMENTUM_SCALE, config.MOMENTUM_MAX_LENGTH)

    # Determine arrow direction (positive dx = moving right, negative = moving left)
    direction = 1 if dx > 0 else -1
    arrow_dx = direction * scaled_length

    # Arrow start point (current hip position)
    start_x = int(current_hip[0]) + x_offset
    start_y = int(current_hip[1])

    # Arrow end point (horizontal only - parallel to ground)
    end_x = int(start_x + arrow_dx)
    end_y = start_y  # Same Y = horizontal/parallel to ground

    # Draw thick arrow with dynamic color
    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        color,
        config.MOMENTUM_THICKNESS,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )


# ============================================================================
# Ball Position Relative to Player (Front/Behind Detection)
# ============================================================================

@dataclass
class BallPositionResult:
    """Result of ball position analysis relative to player."""
    position: str  # "FRONT", "BEHIND", "ALIGNED", or "UNKNOWN"
    movement_direction: Optional[str]  # "LEFT", "RIGHT", or None
    ball_hip_delta_x: float  # ball_x - hip_x (positive = ball right of hip)
    color: Tuple[int, int, int]  # BGR color for visualization


@dataclass
class EdgeZoneStatus:
    """Status of ball relative to screen edges."""
    in_edge_zone: bool           # Ball within 50px of edge
    edge_side: str               # "LEFT", "RIGHT", or "NONE"
    distance_to_edge: float      # Pixels from nearest edge


def determine_ball_position_relative_to_player(
    ball_center: Optional[Tuple[float, float]],
    current_hip: Optional[Tuple[float, float]],
    previous_hip: Optional[Tuple[float, float]],
    config: AnnotationConfig
) -> BallPositionResult:
    """
    Determine if the ball is in front of or behind the player.

    Logic:
    - "FRONT" = ball is in the direction the player is moving
    - "BEHIND" = ball is opposite to player's movement direction
    - "ALIGNED" = ball is directly at player's hip X position (within threshold)

    For Figure-8 drill (horizontal movement):
    - Player moving LEFT (toward Gate 2): FRONT = ball to left of hip
    - Player moving RIGHT (toward Start): FRONT = ball to right of hip

    Args:
        ball_center: (x, y) of ball center in pixels, or None
        current_hip: (x, y) of current hip position, or None
        previous_hip: (x, y) of previous hip position (for direction), or None
        config: Annotation configuration with thresholds

    Returns:
        BallPositionResult with position, direction, delta, and color
    """
    # Handle missing data
    if ball_center is None or current_hip is None:
        return BallPositionResult(
            position="UNKNOWN",
            movement_direction=None,
            ball_hip_delta_x=0.0,
            color=config.BALL_POSITION_NEUTRAL_COLOR
        )

    ball_x = ball_center[0]
    hip_x = current_hip[0]

    # Calculate horizontal distance: positive = ball is to the RIGHT of hip
    delta_x = ball_x - hip_x

    # Determine player movement direction from hip history
    movement_direction: Optional[str] = None
    if previous_hip is not None:
        dx_movement = current_hip[0] - previous_hip[0]
        if dx_movement > config.MOVEMENT_THRESHOLD:
            movement_direction = "RIGHT"  # Moving toward Start cone
        elif dx_movement < -config.MOVEMENT_THRESHOLD:
            movement_direction = "LEFT"   # Moving toward Gate 2
        # else: stationary (None)

    # Check if ball is aligned with player (within threshold)
    if abs(delta_x) < config.BALL_POSITION_THRESHOLD:
        return BallPositionResult(
            position="ALIGNED",
            movement_direction=movement_direction,
            ball_hip_delta_x=delta_x,
            color=config.BALL_POSITION_ALIGNED_COLOR
        )

    # If player is stationary, just report left/right position
    if movement_direction is None:
        # When stationary, can't determine "front" or "behind"
        # Report as neutral with the delta information
        return BallPositionResult(
            position="LEFT" if delta_x < 0 else "RIGHT",
            movement_direction=None,
            ball_hip_delta_x=delta_x,
            color=config.BALL_POSITION_NEUTRAL_COLOR
        )

    # Determine front/behind based on movement direction
    if movement_direction == "LEFT":
        # Player moving left (toward Gate 2 = forward direction in drill)
        # FRONT = ball is to the LEFT of hip (negative delta)
        if delta_x < 0:
            position = "FRONT"
            color = config.BALL_POSITION_FRONT_COLOR
        else:
            position = "BEHIND"
            color = config.BALL_POSITION_BEHIND_COLOR
    else:  # movement_direction == "RIGHT"
        # Player moving right (toward Start = backward direction in drill)
        # FRONT = ball is to the RIGHT of hip (positive delta)
        if delta_x > 0:
            position = "FRONT"
            color = config.BALL_POSITION_FRONT_COLOR
        else:
            position = "BEHIND"
            color = config.BALL_POSITION_BEHIND_COLOR

    return BallPositionResult(
        position=position,
        movement_direction=movement_direction,
        ball_hip_delta_x=delta_x,
        color=color
    )


def check_edge_zone_status(
    ball_x: Optional[float],
    video_width: int,
    config: AnnotationConfig
) -> EdgeZoneStatus:
    """
    Check if ball is in edge zone (matching detection thresholds).

    Args:
        ball_x: Ball center X position in pixels, or None
        video_width: Width of the video frame
        config: Annotation configuration with EDGE_MARGIN

    Returns:
        EdgeZoneStatus with zone info and distance to nearest edge
    """
    if ball_x is None:
        return EdgeZoneStatus(False, "NONE", float('inf'))

    left_distance = ball_x
    right_distance = video_width - ball_x

    # Check right edge (50px danger zone)
    if right_distance < config.EDGE_MARGIN:
        return EdgeZoneStatus(True, "RIGHT", right_distance)

    # Check left edge (50px danger zone)
    if left_distance < config.EDGE_MARGIN:
        return EdgeZoneStatus(True, "LEFT", left_distance)

    return EdgeZoneStatus(False, "NONE", min(left_distance, right_distance))


def draw_ball_position_indicator(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    result: BallPositionResult,
    config: AnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw visual indicators showing ball position relative to player.

    Visualizes:
    1. Vertical dividing line through player's hip (the decision boundary)
    2. Connecting line from hip to ball (colored by front/behind status)
    3. "FRONT"/"BEHIND" label near the ball

    Args:
        frame: Video frame to draw on
        ball_center: (x, y) of ball center in pixels
        hip_position: (x, y) of hip in pixels
        result: BallPositionResult from detection
        config: Annotation configuration
        x_offset: Horizontal offset for sidebar
    """
    if ball_center is None or hip_position is None:
        return

    if result.position == "UNKNOWN":
        return

    hip_x = int(hip_position[0]) + x_offset
    hip_y = int(hip_position[1])
    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # 1. Draw vertical dividing line through hip (the decision boundary)
    # This shows WHERE the front/behind boundary is
    half_height = config.DIVIDER_LINE_HEIGHT // 2
    line_color = (100, 100, 100)  # Gray dashed line

    # Draw dashed vertical line
    dash_length = 10
    gap_length = 5
    y_start = hip_y - half_height
    y_end = hip_y + half_height
    y = y_start
    while y < y_end:
        y_next = min(y + dash_length, y_end)
        cv2.line(frame, (hip_x, y), (hip_x, y_next), line_color, 1, cv2.LINE_AA)
        y = y_next + gap_length

    # Draw small marker at hip position
    cv2.circle(frame, (hip_x, hip_y), 5, result.color, -1)
    cv2.circle(frame, (hip_x, hip_y), 5, (255, 255, 255), 1)

    # 2. Draw connecting line from hip to ball (shows the relationship)
    cv2.line(frame, (hip_x, hip_y), (ball_x, ball_y),
             result.color, config.BALL_HIP_LINE_THICKNESS, cv2.LINE_AA)

    # 3. Draw position label near the ball
    label = result.position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    # Position label above the ball
    label_x = ball_x - text_width // 2
    label_y = ball_y - 30  # Above the ball

    # Background for label
    padding = 3
    cv2.rectangle(
        frame,
        (label_x - padding, label_y - text_height - padding),
        (label_x + text_width + padding, label_y + padding),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, label,
        (label_x, label_y),
        font, font_scale, result.color, font_thickness, cv2.LINE_AA
    )

    # 4. Draw movement direction arrow indicator at top of divider line
    if result.movement_direction:
        arrow_y = y_start - 15
        arrow_length = 25
        if result.movement_direction == "LEFT":
            arrow_end_x = hip_x - arrow_length
        else:
            arrow_end_x = hip_x + arrow_length

        cv2.arrowedLine(
            frame,
            (hip_x, arrow_y),
            (arrow_end_x, arrow_y),
            result.color, 2, tipLength=0.4, line_type=cv2.LINE_AA
        )


def draw_behind_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    config: AnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw ball-behind duration counter on video.

    Shows how many consecutive frames the ball has been behind the player.
    Red when actively behind, yellow when showing persisted value after.

    Args:
        frame: Video frame to draw on
        count: Number of consecutive behind frames
        is_active: True if currently behind, False if showing persist value
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    if count <= 0:
        return

    text = f"BEHIND: {count}f"
    x = x_offset + config.BEHIND_COUNTER_POS_X
    y = config.BEHIND_COUNTER_POS_Y

    # Use red if active, yellow if persisting
    color = config.BEHIND_COUNTER_COLOR if is_active else config.BEHIND_COUNTER_PERSIST_COLOR

    # Get text size for background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, config.BEHIND_COUNTER_FONT_SCALE, 2)

    # Draw background box
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                config.BEHIND_COUNTER_FONT_SCALE, color, 2, cv2.LINE_AA)


def draw_edge_zones(
    frame: np.ndarray,
    video_width: int,
    video_height: int,
    config: AnnotationConfig,
    x_offset: int = 0
) -> np.ndarray:
    """
    Draw semi-transparent edge zone overlays on frame (50px danger zones).

    Args:
        frame: Video frame to draw on
        video_width: Original video width (before sidebar)
        video_height: Video height
        config: Annotation configuration
        x_offset: Horizontal offset for sidebar

    Returns:
        Frame with edge zone overlays blended in
    """
    overlay = frame.copy()

    # Draw danger zones (50px from edges) - Red, subtle 15% opacity
    # Left edge danger zone
    cv2.rectangle(
        overlay,
        (x_offset, 0),
        (x_offset + config.EDGE_MARGIN, video_height),
        config.EDGE_ZONE_COLOR, -1
    )
    # Right edge danger zone
    cv2.rectangle(
        overlay,
        (x_offset + video_width - config.EDGE_MARGIN, 0),
        (x_offset + video_width, video_height),
        config.EDGE_ZONE_COLOR, -1
    )

    # Blend with original frame (15% opacity)
    return cv2.addWeighted(overlay, config.EDGE_ZONE_ALPHA, frame,
                           1 - config.EDGE_ZONE_ALPHA, 0)


def draw_edge_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    edge_side: str,
    config: AnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw edge zone frame counter (similar to behind counter).

    Shows how many consecutive frames the ball has been in the edge zone.
    Red when actively in zone, yellow when showing persisted value.

    Args:
        frame: Video frame to draw on
        count: Number of consecutive frames in edge zone
        is_active: True if currently in zone, False if showing persist value
        edge_side: Which edge ("LEFT" or "RIGHT")
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    if count <= 0:
        return

    text = f"EDGE ({edge_side}): {count}f"
    x = x_offset + config.EDGE_COUNTER_POS_X
    y = config.EDGE_COUNTER_POS_Y

    # Use red if active, yellow if persisting
    color = config.EDGE_COUNTER_COLOR if is_active else config.EDGE_COUNTER_PERSIST_COLOR

    # Get text size for background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, config.EDGE_COUNTER_FONT_SCALE, 2)

    # Draw background box
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                config.EDGE_COUNTER_FONT_SCALE, color, 2, cv2.LINE_AA)


# ============================================================================
# Video Processing
# ============================================================================

def annotate_video_with_json_cones(video_path: Path, parquet_dir: Path, output_path: Path,
                                   config: AnnotationConfig = None) -> bool:
    """
    Annotate video with JSON cone positions (static) and parquet ball/pose (dynamic).
    Adds a sidebar on the left showing all object coordinates.

    Args:
        video_path: Path to input video
        parquet_dir: Directory containing parquet files and cone_annotations.json
        output_path: Path for output annotated video
        config: Annotation configuration

    Returns:
        True if successful
    """
    if config is None:
        config = AnnotationConfig()

    base_name = parquet_dir.name

    # Load JSON cone annotations
    json_path = parquet_dir / "cone_annotations.json"
    if not json_path.exists():
        print(f"  Error: cone_annotations.json not found in {parquet_dir}")
        return False

    print(f"  Loading JSON cone annotations...")
    layout = Figure8Layout.from_json(json_path)
    print(f"    Gate 1 width: {layout.gate1_width:.0f}px")
    print(f"    Gate 2 width: {layout.gate2_width:.0f}px")

    # Create turning zones from layout
    print(f"  Creating turning zones...")
    zone_config = TurningZoneConfig.default()
    turning_zones = create_turning_zones(layout, zone_config)
    print(f"    START zone: center=({turning_zones.start_zone.center_px:.0f}, {turning_zones.start_zone.center_py:.0f}), "
          f"axes=({turning_zones.start_zone.semi_major:.0f}, {turning_zones.start_zone.semi_minor:.0f})")
    print(f"    GATE2 zone: center=({turning_zones.gate2_zone.center_px:.0f}, {turning_zones.gate2_zone.center_py:.0f}), "
          f"axes=({turning_zones.gate2_zone.semi_major:.0f}, {turning_zones.gate2_zone.semi_minor:.0f})")

    # Load parquet data
    ball_path = parquet_dir / f"{base_name}_football.parquet"
    pose_path = parquet_dir / f"{base_name}_pose.parquet"

    if not ball_path.exists() or not pose_path.exists():
        print(f"  Error: Missing parquet files in {parquet_dir}")
        return False

    print(f"  Loading parquet data...")
    ball_df = load_ball_data(ball_path)
    pose_df = load_pose_data(pose_path)

    # Create lookup structures
    ball_lookup = ball_df.groupby('frame_id').apply(
        lambda g: g[['x1', 'y1', 'x2', 'y2', 'confidence']].to_dict('records')
    ).to_dict()
    pose_lookup = prepare_pose_lookup(pose_df)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video: {video_path}")
        return False

    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # New canvas dimensions (sidebar + video)
    canvas_width = config.SIDEBAR_WIDTH + orig_width
    canvas_height = orig_height

    print(f"  Video: {orig_width}x{orig_height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"  Output canvas: {canvas_width}x{canvas_height} (sidebar: {config.SIDEBAR_WIDTH}px)")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer with expanded canvas
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_width, canvas_height))

    if not out.isOpened():
        print(f"  Error: Cannot create output video: {output_path}")
        cap.release()
        return False

    # Process frames
    print(f"  Processing frames...")

    # Track hip positions for momentum calculation (10-frame lookback)
    hip_history: deque = deque(maxlen=config.MOMENTUM_LOOKBACK_FRAMES + 1)

    # Ball-behind duration tracking (debug feature)
    behind_counter: int = 0           # Consecutive frames ball has been behind
    behind_display_value: int = 0     # Value to display (preserved after behind ends)
    behind_display_timer: int = 0     # Frames remaining to show the count
    behind_persist_frames = int(config.BEHIND_COUNTER_PERSIST_SECONDS * fps)

    # Edge zone tracking (similar to behind counter)
    edge_counter: int = 0                    # Current consecutive frames in edge zone
    edge_display_value: int = 0              # Value to show (preserved after exit)
    edge_display_timer: int = 0              # Frames remaining to show count
    edge_last_side: str = "NONE"             # Which edge (LEFT/RIGHT)
    edge_persist_frames = int(config.EDGE_COUNTER_PERSIST_SECONDS * fps)

    # Drill event tracker - detects cone crossings at START cone
    # Pass bbox y2 (bottom edge) for accurate above/below threshold
    drill_tracker = DrillEventTracker(
        start_cone_x=layout.start.px,
        start_cone_y=layout.start.py,
        start_cone_y2=layout.start.y2  # Bottom edge of cone bbox
    )
    print(f"  Drill tracker initialized at START cone ({layout.start.px}, {layout.start.py}), y2_threshold={layout.start.y2}")

    for frame_id in tqdm(range(total_frames), desc="  Annotating", unit="frame"):
        ret, video_frame = cap.read()
        if not ret:
            break

        # Create expanded canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place video frame on the right side (after sidebar)
        canvas[:, config.SIDEBAR_WIDTH:] = video_frame

        # Get current ball data
        balls = ball_lookup.get(frame_id, [])
        ball_center = None
        ball_bbox_for_tracker = None
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                center_x = (ball['x1'] + ball['x2']) / 2
                center_y = (ball['y1'] + ball['y2']) / 2
                ball_center = (center_x, center_y)
                ball_bbox_for_tracker = ball  # Keep full bbox for event tracking
                break  # Use first valid ball

        # Update drill event tracker (detect cone crossings)
        drill_tracker.update(frame_id, ball_bbox_for_tracker, fps)

        # Get current pose keypoints (use first person)
        persons = pose_lookup.get(frame_id, {})
        pose_keypoints = {}
        if persons:
            first_person_id = min(persons.keys())
            pose_keypoints = persons[first_person_id]

        # Get current hip position from first valid person (used for multiple features)
        current_hip = None
        for _, keypoints in persons.items():
            hip_data = keypoints.get('hip')
            if hip_data and hip_data[2] >= config.MIN_KEYPOINT_CONFIDENCE:
                current_hip = (hip_data[0], hip_data[1])
                break

        # Update hip history (needed for direction detection)
        if current_hip:
            hip_history.append(current_hip)

        # Get previous hip position for direction detection
        previous_hip = hip_history[0] if len(hip_history) >= 2 else None

        # DETECTION: Determine ball position relative to player (FRONT/BEHIND)
        ball_position_result = None
        if config.DRAW_BALL_POSITION:
            ball_position_result = determine_ball_position_relative_to_player(
                ball_center=ball_center,
                current_hip=current_hip,
                previous_hip=previous_hip,
                config=config
            )

        # Update ball-behind duration counter
        if ball_position_result and ball_position_result.position == "BEHIND":
            behind_counter += 1
            behind_display_value = behind_counter
            behind_display_timer = behind_persist_frames  # Reset timer
        else:
            if behind_counter > 0:
                # Just ended - start persist timer with final value
                behind_display_value = behind_counter
                behind_display_timer = behind_persist_frames
            behind_counter = 0  # Reset counter

        # Decrement display timer
        if behind_display_timer > 0:
            behind_display_timer -= 1

        # Edge zone detection and counter update
        ball_center_x = ball_center[0] if ball_center else None
        edge_status = check_edge_zone_status(ball_center_x, orig_width, config)

        if edge_status.in_edge_zone:
            # Check if switched sides (e.g., detection glitch causing jump)
            if edge_last_side != "NONE" and edge_last_side != edge_status.edge_side:
                # Side changed - reset counter (treat as new edge event)
                edge_counter = 1
            else:
                edge_counter += 1
            edge_display_value = edge_counter
            edge_last_side = edge_status.edge_side
            edge_display_timer = edge_persist_frames  # Reset persist timer
        else:
            if edge_counter > 0:
                # Just exited edge zone - start persist timer
                edge_display_timer = edge_persist_frames
            edge_counter = 0
            if edge_display_timer > 0:
                edge_display_timer -= 1

        # STATIC: Draw edge zones first (underneath other elements) - subtle red 50px margins
        canvas = draw_edge_zones(
            canvas, orig_width, orig_height, config,
            x_offset=config.SIDEBAR_WIDTH
        )

        # STATIC: Draw turning zones (underneath other elements)
        active_zone = draw_turning_zones(
            canvas, turning_zones, ball_center,
            x_offset=config.SIDEBAR_WIDTH,
            start_color=config.START_ZONE_COLOR,
            gate2_color=config.GATE2_ZONE_COLOR,
            highlight_color=config.ZONE_HIGHLIGHT_COLOR,
            alpha=config.ZONE_ALPHA,
        )

        # Draw sidebar with coordinates (including zone status, ball position, and drill events)
        draw_sidebar(canvas, frame_id, layout, ball_center, pose_keypoints, config,
                     active_zone, ball_position_result,
                     drill_events=drill_tracker.get_recent_events(config.EVENT_LOG_MAX_EVENTS))

        # STATIC: Draw JSON cones on video area (with x offset)
        draw_json_cones(canvas, layout, config, x_offset=config.SIDEBAR_WIDTH)

        # STATIC: Draw threshold lines for pass above/below debug
        # Vertical magenta line = X crossing threshold (start cone px)
        # Horizontal green line = Y above/below threshold (start cone y2)
        if config.DRAW_THRESHOLD_LINES:
            draw_cone_threshold_lines(
                canvas,
                cone_x=layout.start.px,
                cone_y_threshold=layout.start.y2,
                video_width=orig_width,
                video_height=orig_height,
                x_offset=config.SIDEBAR_WIDTH,
                x_color=config.THRESHOLD_X_COLOR,
                y_color=config.THRESHOLD_Y_COLOR,
                thickness=config.THRESHOLD_LINE_THICKNESS,
            )

        # DYNAMIC: Draw ball from parquet (coordinates shown in sidebar)
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                label = f"Ball {ball['confidence']:.2f}"
                draw_bbox(canvas, ball['x1'], ball['y1'], ball['x2'], ball['y2'],
                         config.BALL_COLOR, label, config, x_offset=config.SIDEBAR_WIDTH)

        # DYNAMIC: Draw debug axes from ball bottom position (for cone crossing debug)
        # Horizontal line is YELLOW if ball above cone, RED if below cone
        if config.DRAW_DEBUG_AXES and ball_bbox_for_tracker:
            ball_bottom_x = (ball_bbox_for_tracker['x1'] + ball_bbox_for_tracker['x2']) / 2
            ball_bottom_y = ball_bbox_for_tracker['y2']  # Bottom = ground contact
            draw_debug_axes(
                canvas,
                ball_bottom_x=ball_bottom_x,
                ball_bottom_y=ball_bottom_y,
                video_width=orig_width,
                video_height=orig_height,
                x_offset=config.SIDEBAR_WIDTH,
                color=config.DEBUG_AXES_COLOR,
                thickness=config.DEBUG_AXES_THICKNESS,
                cone_y=layout.start.py,  # Pass cone Y for reference
                color_below=(0, 0, 255),  # Red when below cone
                cone_y_threshold=layout.start.y2,  # Use bbox bottom edge for above/below check
            )

        # DYNAMIC: Draw pose skeletons from parquet
        for _, keypoints in persons.items():
            draw_skeleton(canvas, keypoints, config, x_offset=config.SIDEBAR_WIDTH)

        # DYNAMIC: Draw ball position indicator (shows front/behind logic visualization)
        if config.DRAW_BALL_POSITION and ball_position_result:
            draw_ball_position_indicator(
                canvas, ball_center, current_hip, ball_position_result,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # DYNAMIC: Draw momentum arrow from hip movement
        if config.DRAW_MOMENTUM_ARROW and current_hip and previous_hip:
            draw_momentum_arrow(
                canvas, current_hip, previous_hip,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # DYNAMIC: Draw ball-behind duration counter (debug feature)
        if behind_display_timer > 0 or behind_counter > 0:
            draw_behind_counter(
                canvas,
                behind_display_value,
                is_active=(behind_counter > 0),
                config=config,
                x_offset=config.SIDEBAR_WIDTH
            )

        # DYNAMIC: Draw edge zone counter (debug feature)
        if edge_display_timer > 0 or edge_counter > 0:
            draw_edge_counter(
                canvas,
                edge_display_value,
                is_active=(edge_counter > 0),
                edge_side=edge_last_side,
                config=config,
                x_offset=config.SIDEBAR_WIDTH
            )

        out.write(canvas)

    cap.release()
    out.release()

    print(f"  Saved (mp4v): {output_path}")

    # Convert to H.264 for better compatibility
    h264_path = convert_to_h264(output_path)
    if h264_path:
        return True
    else:
        print(f"  Warning: H.264 conversion failed, keeping mp4v version")
        return True


def convert_to_h264(input_path: Path) -> Path:
    """
    Convert video to H.264 codec using ffmpeg for better compatibility.

    Args:
        input_path: Path to input mp4v video

    Returns:
        Path to H.264 video, or None if conversion failed
    """
    # Create temporary output path
    temp_path = input_path.parent / f"{input_path.stem}_h264_temp.mp4"

    print(f"  Converting to H.264 for compatibility...")

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',  # Faster encoding (was 'medium')
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(temp_path)
    ]

    try:
        # Timeout after 5 minutes to prevent hanging
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and temp_path.exists():
            # Replace original with H.264 version
            backup_path = input_path.parent / f"{input_path.stem}_mp4v.mp4"
            input_path.rename(backup_path)
            temp_path.rename(input_path)
            print(f"  Converted to H.264: {input_path}")
            # Remove backup (mp4v version)
            if backup_path.exists():
                backup_path.unlink()
                print(f"  Removed mp4v backup")
            return input_path
        else:
            print(f"  FFmpeg error: {result.stderr}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    except FileNotFoundError:
        print(f"  FFmpeg not found - keeping mp4v format")
        return None
    except subprocess.TimeoutExpired:
        print(f"  FFmpeg timeout (>5min) - keeping mp4v format")
        if temp_path.exists():
            temp_path.unlink()
        return None
    except Exception as e:
        print(f"  Conversion error: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None


# ============================================================================
# Main
# ============================================================================

def get_available_videos(videos_dir: Path, parquet_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Get list of videos with matching parquet data and JSON annotations."""
    available = []

    for parquet_path in sorted(parquet_dir.iterdir()):
        if not parquet_path.is_dir():
            continue

        base_name = parquet_path.name
        json_path = parquet_path / "cone_annotations.json"

        if not json_path.exists():
            continue

        # Look for matching video
        video_path = videos_dir / f"{base_name}.MOV"
        if video_path.exists():
            available.append((base_name, video_path, parquet_path))

    return available


def main():
    parser = argparse.ArgumentParser(
        description="Annotate F8 drill videos with JSON cone annotations and coordinate sidebar"
    )
    parser.add_argument(
        "video_name",
        nargs="?",
        help="Name of video to process (e.g., abdullah_nasib_f8)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available videos with JSON annotations"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process ALL available videos with JSON annotations"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have JSON cone annotated output"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/videos"),
        help="Directory containing source videos"
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/video_detection_pose_ball_cones"),
        help="Directory containing parquet data folders"
    )

    args = parser.parse_args()

    available = get_available_videos(args.videos_dir, args.parquet_dir)

    if args.list:
        print("\n Available videos with JSON cone annotations:\n")
        for name, video_path, parquet_path in available:
            output_path = parquet_path / f"{name}_json_cones.mp4"
            status = "[done]" if output_path.exists() else "[    ]"
            print(f"  {status} {name}")
        print(f"\nTotal: {len(available)} videos")
        print("([done] = already processed with sidebar)")
        return 0

    if args.all:
        # Process all available videos
        print(f"\n Processing ALL {len(available)} videos with JSON cone annotations + sidebar...\n")
        config = AnnotationConfig()

        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, (name, video_path, parquet_path) in enumerate(available, 1):
            output_path = parquet_path / f"{name}_json_cones.mp4"

            # Skip if already exists and flag is set
            if args.skip_existing and output_path.exists():
                print(f"[{i}/{len(available)}] Skipping {name} (already exists)")
                skip_count += 1
                continue

            print(f"\n{'='*60}")
            print(f"[{i}/{len(available)}] Processing: {name}")
            print(f"{'='*60}")

            success = annotate_video_with_json_cones(video_path, parquet_path, output_path, config)

            if success:
                success_count += 1
                print(f"  Done: {output_path.name}")
            else:
                fail_count += 1
                print(f"  Failed: {name}")

        print(f"\n{'='*60}")
        print(f" BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed: {success_count}")
        print(f"  Skipped:   {skip_count}")
        print(f"  Failed:    {fail_count}")
        print(f"  Total:     {len(available)}")

        return 0 if fail_count == 0 else 1

    if not args.video_name:
        print("Error: Please specify a video name, use --list, or use --all")
        return 1

    # Find matching video
    to_process = None
    for name, video_path, parquet_path in available:
        if name == args.video_name or name.startswith(args.video_name):
            to_process = (name, video_path, parquet_path)
            break

    if not to_process:
        print(f"Error: Video not found: {args.video_name}")
        print("Use --list to see available videos")
        return 1

    name, video_path, parquet_path = to_process

    print(f"\n Annotating {name} with JSON cones + coordinate sidebar...\n")

    # Output with different name to preserve original
    output_path = parquet_path / f"{name}_json_cones.mp4"

    config = AnnotationConfig()
    success = annotate_video_with_json_cones(video_path, parquet_path, output_path, config)

    if success:
        print(f"\n Done! Output: {output_path}")
        return 0
    else:
        print("\n Annotation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
