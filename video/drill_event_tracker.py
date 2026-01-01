#!/usr/bin/env python3
"""
Drill Event Tracker for Figure-8 Cone Drill Analysis.

Tracks ball crossings at the START cone X position and detects:
- PASS_BEHIND: Ball crosses cone X while above/behind the cone (ball_y < cone_y)
- PASS_FRONT: Ball crosses cone X while below/in front of cone (ball_y > cone_y)
- TURN: Ball returns from opposite direction after a crossing

Usage:
    tracker = DrillEventTracker(start_cone_x=1593, start_cone_y=878, start_cone_y2=890)

    for frame_id, ball_bbox, fps in frames:
        event = tracker.update(frame_id, ball_bbox, fps)

    events = tracker.get_events()
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np


class DrillEventType(Enum):
    """Types of drill events at the START cone."""
    PASS_ABOVE = "PASS_ABOVE"    # Ball crosses cone X while ball_y < cone_y (above cone in screen coords)
    PASS_BELOW = "PASS_BELOW"    # Ball crosses cone X while ball_y > cone_y (below cone in screen coords)
    TURN = "TURN"                 # Sequence of opposite crossings (above->below or below->above)


@dataclass
class DrillEvent:
    """A single drill event at the START cone."""
    event_type: DrillEventType
    timestamp_seconds: float      # Time in seconds
    frame_id: int                 # Frame number when event occurred
    ball_x: float                 # Ball X position at crossing
    ball_y: float                 # Ball Y position at crossing (y2 = bottom)
    ball_above_cone: bool         # True if ball_y < cone_y (above/behind)

    def format_timestamp(self) -> str:
        """Format timestamp as MM:SS.d"""
        minutes = int(self.timestamp_seconds // 60)
        seconds = self.timestamp_seconds % 60
        return f"{minutes:02d}:{seconds:04.1f}"


class DrillEventTracker:
    """
    Tracks ball crossings at the START cone X position.

    Detects:
    - PASS_BEHIND: Ball crosses cone X while above cone (ball_y < cone_y)
    - PASS_FRONT: Ball crosses cone X while below cone (ball_y > cone_y)
    - TURN: Ball returns from opposite direction after a PASS event

    Coordinate system:
    - Y increases DOWNWARD (OpenCV convention)
    - ball_y < cone_y means ball is "above" (further from camera / behind cone)
    - ball_y > cone_y means ball is "below" (closer to camera / in front of cone)

    Usage:
        tracker = DrillEventTracker(start_cone_x=1593, start_cone_y=878, start_cone_y2=890)

        for frame_id, ball_bbox, fps in frames:
            tracker.update(frame_id, ball_bbox, fps)

        events = tracker.get_events()
    """

    def __init__(self, start_cone_x: float, start_cone_y: float, start_cone_y2: Optional[float] = None):
        """
        Initialize tracker with START cone position.

        Args:
            start_cone_x: X coordinate of start cone center (pixels)
            start_cone_y: Y coordinate of start cone center (pixels)
            start_cone_y2: Bottom edge (y2) of cone bounding box (pixels).
                          If provided, used for above/below threshold instead of center.
        """
        self.start_cone_x = start_cone_x
        self.start_cone_y = start_cone_y
        # Use bbox bottom edge for above/below check, fallback to center if not provided
        self.start_cone_y_threshold = start_cone_y2 if start_cone_y2 is not None else start_cone_y

        # Event tracking
        self._events: List[DrillEvent] = []

        # State tracking
        self._prev_ball_x: Optional[float] = None
        self._prev_ball_above: Optional[bool] = None  # True if ball was above cone
        self._last_crossing_above: Optional[bool] = None  # Last PASS event direction

    def update(
        self,
        frame_id: int,
        ball_bbox: Optional[Dict[str, Any]],
        fps: float
    ) -> Optional[DrillEvent]:
        """
        Update tracker with new frame data.

        Args:
            frame_id: Current frame number
            ball_bbox: Ball bounding box dict with x1, y1, x2, y2, or None if no ball
            fps: Video FPS for timestamp calculation

        Returns:
            DrillEvent if an event was detected, else None
        """
        if ball_bbox is None:
            return None

        # Use ball center X, bottom Y (y2) as ground contact reference
        ball_x = (ball_bbox['x1'] + ball_bbox['x2']) / 2  # Center X
        ball_y = ball_bbox['y2']  # Bottom edge = ground contact

        timestamp = frame_id / fps

        # Determine if ball is above or below cone
        # Y increases downward, so ball_y < cone_y means ball is ABOVE (behind)
        # Uses bbox bottom edge (y2) as threshold - ball must be above the entire cone box
        ball_above_cone = ball_y < self.start_cone_y_threshold

        event = None

        # Check for crossing (ball X crosses cone X)
        if self._prev_ball_x is not None:
            crossed = self._check_crossing(self._prev_ball_x, ball_x)

            if crossed:
                # Determine event type based on ball Y relative to cone
                # Y increases downward, so ball_y < cone_y means ball is ABOVE
                if ball_above_cone:
                    pass_type = DrillEventType.PASS_ABOVE
                else:
                    pass_type = DrillEventType.PASS_BELOW

                # Create PASS event
                pass_event = DrillEvent(
                    event_type=pass_type,
                    timestamp_seconds=timestamp,
                    frame_id=frame_id,
                    ball_x=ball_x,
                    ball_y=ball_y,
                    ball_above_cone=ball_above_cone
                )
                self._events.append(pass_event)
                event = pass_event

                # Check for TURN (opposite crossing from last)
                if self._last_crossing_above is not None:
                    if self._last_crossing_above != ball_above_cone:
                        # Direction changed = TURN event
                        turn_event = DrillEvent(
                            event_type=DrillEventType.TURN,
                            timestamp_seconds=timestamp,
                            frame_id=frame_id,
                            ball_x=ball_x,
                            ball_y=ball_y,
                            ball_above_cone=ball_above_cone
                        )
                        self._events.append(turn_event)
                        event = turn_event  # Return TURN as primary event

                # Update last crossing direction
                self._last_crossing_above = ball_above_cone

        # Update state for next frame
        self._prev_ball_x = ball_x
        self._prev_ball_above = ball_above_cone

        return event

    def _check_crossing(self, prev_x: float, curr_x: float) -> bool:
        """
        Check if ball crossed the cone X position between frames.

        Args:
            prev_x: Previous frame ball X
            curr_x: Current frame ball X

        Returns:
            True if ball crossed cone X line
        """
        # Ball crossed if it was on one side and now on the other
        was_left = prev_x < self.start_cone_x
        now_left = curr_x < self.start_cone_x
        return was_left != now_left

    def get_events(self) -> List[DrillEvent]:
        """Get all detected events."""
        return self._events.copy()

    def get_recent_events(self, n: int = 10) -> List[DrillEvent]:
        """
        Get the N most recent events.

        Args:
            n: Number of events to return

        Returns:
            List of most recent events (newest last)
        """
        return self._events[-n:] if len(self._events) > n else self._events.copy()

    def reset(self):
        """Reset tracker state."""
        self._events = []
        self._prev_ball_x = None
        self._prev_ball_above = None
        self._last_crossing_above = None


# ============================================================================
# Drawing Functions
# ============================================================================

def draw_debug_axes(
    canvas: np.ndarray,
    ball_bottom_x: float,
    ball_bottom_y: float,
    video_width: int,
    video_height: int,
    x_offset: int = 0,
    color: Tuple[int, int, int] = (0, 255, 255),  # Yellow BGR
    thickness: int = 1,
    cone_y: Optional[float] = None,
    color_below: Tuple[int, int, int] = (0, 0, 255),  # Red BGR for below cone
    cone_y_threshold: Optional[float] = None,  # Bbox y2 for above/below check
) -> None:
    """
    Draw full-frame reference lines from ball bottom position.

    Draws:
    - Vertical line from ball X extending top to bottom of frame (always yellow)
    - Horizontal line from ball Y extending left to right of frame
      - Yellow if ball is ABOVE cone (ball_y < threshold)
      - Red if ball is BELOW cone (ball_y >= threshold)

    Args:
        canvas: Full canvas (sidebar + video)
        ball_bottom_x: Ball bottom X position (center X)
        ball_bottom_y: Ball bottom Y position (y2)
        video_width: Original video width
        video_height: Video height
        x_offset: Sidebar offset (typically 300)
        color: BGR color for vertical line and horizontal when above cone
        thickness: Line thickness
        cone_y: Y position of cone center (for drawing reference)
        color_below: BGR color for horizontal line when ball is below cone
        cone_y_threshold: Threshold Y for above/below check (bbox y2). Uses cone_y if not provided.
    """
    # Adjust for sidebar offset
    draw_x = int(ball_bottom_x) + x_offset
    draw_y = int(ball_bottom_y)

    # Vertical line through ball X (full frame height) - always same color
    cv2.line(canvas, (draw_x, 0), (draw_x, video_height), color, thickness)

    # Horizontal line through ball Y - color depends on above/below cone
    # Use cone_y_threshold (bbox y2) if provided, else fall back to cone_y (center)
    threshold = cone_y_threshold if cone_y_threshold is not None else cone_y
    if threshold is not None and ball_bottom_y >= threshold:
        # Ball is BELOW cone (at or below bbox bottom edge) - use red
        h_color = color_below
    else:
        # Ball is ABOVE cone (or no threshold provided) - use yellow
        h_color = color

    cv2.line(canvas, (x_offset, draw_y), (x_offset + video_width, draw_y), h_color, thickness)


def draw_cone_threshold_lines(
    canvas: np.ndarray,
    cone_x: float,
    cone_y_threshold: float,
    video_width: int,
    video_height: int,
    x_offset: int = 0,
    x_color: Tuple[int, int, int] = (255, 0, 255),  # Magenta BGR for X threshold
    y_color: Tuple[int, int, int] = (0, 255, 0),    # Green BGR for Y threshold
    thickness: int = 2
) -> None:
    """
    Draw static reference lines at cone threshold positions for pass above/below debug.

    These lines show the exact thresholds used for PASS_ABOVE/PASS_BELOW detection:
    - Vertical line at cone_x: When ball crosses this X, a PASS event is triggered
    - Horizontal line at cone_y_threshold: Determines if PASS_ABOVE or PASS_BELOW

    Args:
        canvas: Full canvas (sidebar + video)
        cone_x: X coordinate threshold (start cone px)
        cone_y_threshold: Y coordinate threshold (start cone y2 / bbox bottom)
        video_width: Original video width
        video_height: Video height
        x_offset: Sidebar offset
        x_color: BGR color for vertical X threshold line
        y_color: BGR color for horizontal Y threshold line
        thickness: Line thickness
    """
    draw_x = int(cone_x) + x_offset
    draw_y = int(cone_y_threshold)

    # Vertical line at cone X (crossing detection threshold) - full height
    cv2.line(canvas, (draw_x, 0), (draw_x, video_height), x_color, thickness, cv2.LINE_AA)

    # Horizontal line at cone Y threshold (above/below) - full width across video
    cv2.line(canvas, (x_offset, draw_y), (x_offset + video_width, draw_y), y_color, thickness, cv2.LINE_AA)


# Color coding for event types (BGR format)
EVENT_COLORS = {
    DrillEventType.PASS_ABOVE: (0, 100, 255),    # Orange-red (ball above cone)
    DrillEventType.PASS_BELOW: (0, 255, 100),    # Green (ball below cone)
    DrillEventType.TURN: (255, 255, 0),           # Cyan (direction changed)
}


def draw_event_log(
    canvas: np.ndarray,
    events: List[DrillEvent],
    start_y: int,
    sidebar_width: int = 300,
    sidebar_padding: int = 12,
    line_height: int = 24,
    font_scale: float = 0.55,
    header_color: Tuple[int, int, int] = (80, 80, 80),
    max_events: int = 8
) -> int:
    """
    Draw event log in the sidebar.

    Shows recent events with timestamps and types.

    Args:
        canvas: Full canvas to draw on
        events: List of DrillEvent objects
        start_y: Y position to start drawing
        sidebar_width: Width of sidebar
        sidebar_padding: Padding from edge
        line_height: Height per line
        font_scale: Font scale
        header_color: Header line color
        max_events: Maximum events to display

    Returns:
        Y position after drawing (for next section)
    """
    y = start_y

    # Section header line
    cv2.line(canvas, (sidebar_padding, y),
             (sidebar_width - sidebar_padding, y),
             header_color, 1)
    y += 18

    # Section title
    cv2.putText(canvas, "DRILL EVENTS", (sidebar_padding, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += 8

    # Show recent events (newest at top)
    recent = events[-max_events:] if len(events) > max_events else events

    if not recent:
        y += line_height
        cv2.putText(canvas, "  No events yet", (sidebar_padding, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1,
                    (100, 100, 100), 1, cv2.LINE_AA)
        return y + line_height

    for event in reversed(recent):  # Newest first
        y += line_height

        # Format: "00:05.2  TURN"
        time_str = event.format_timestamp()
        type_str = event.event_type.value

        color = EVENT_COLORS.get(event.event_type, (200, 200, 200))

        # Draw timestamp (gray)
        cv2.putText(canvas, time_str, (sidebar_padding + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # Draw event type (colored)
        cv2.putText(canvas, type_str, (sidebar_padding + 75, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.05,
                    color, 1, cv2.LINE_AA)

    return y + line_height
