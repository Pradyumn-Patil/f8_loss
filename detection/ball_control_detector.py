"""
Ball Control Detector - Simplified for Figure-8 drill.

Detects ball control loss events during Figure-8 cone drill.
Tracks gate passages, drill phases, and lap completion.

DETECTION LOGIC LOCATION:
========================
All loss detection logic is in the `detect_loss()` method.
Modify ONLY that method to implement your detection algorithm.

Current implementation detects:
1. BOUNDARY_VIOLATION - Ball exits video frame
2. BALL_BEHIND_PLAYER - Ball stays behind player for sustained period
"""
import logging
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import AppConfig, Figure8DrillConfig
from .data_structures import (
    ControlState, EventType, FrameData,
    LossEvent, DetectionResult,
    DrillPhase, DrillDirection, GatePassage, ConeRole, Figure8Layout
)
from .data_loader import (
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    load_cone_annotations
)
from .figure8_cone_detector import Figure8ConeDetector
from .turning_zones import TurningZoneSet, TurningZoneConfig, create_turning_zones

logger = logging.getLogger(__name__)


class BallControlDetector:
    """
    Main class for detecting ball control loss events in Figure-8 drill.

    Usage:
        config = AppConfig.for_figure8()
        detector = BallControlDetector(config)
        result = detector.detect(ball_df, pose_df, cone_df)
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        parquet_dir: Optional[str] = None,
        video_path: Optional[str] = None
    ):
        """
        Initialize detector with Figure-8 config.

        Args:
            config: Application configuration (defaults to Figure-8 config)
            parquet_dir: Path to parquet directory for loading manual cone annotations
            video_path: Path to video file for getting actual video dimensions
        """
        self.config = config or AppConfig.for_figure8()
        self.parquet_dir = parquet_dir
        self.video_path = video_path

        if not isinstance(self.config.drill, Figure8DrillConfig):
            self.config.drill = Figure8DrillConfig()

        self._detection_config = self.config.detection
        self._drill_config: Figure8DrillConfig = self.config.drill

        # Figure-8 specific detector
        self._f8_detector: Optional[Figure8ConeDetector] = None

        # Cone roles from JSON annotations (static positions)
        self._cone_roles: List[ConeRole] = []

        # Video dimensions (will be populated from video file)
        self._video_width: int = 1920  # Default fallback
        self._video_height: int = 1080
        if video_path:
            self._load_video_dimensions(video_path)

        # State tracking
        self._current_state = ControlState.CONTROLLED
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = DrillPhase.AT_START
        self._events: List[LossEvent] = []
        self._frame_data: List[FrameData] = []
        self._event_counter = 0

        # Previous frame data
        self._prev_ball_pos: Optional[Tuple[float, float]] = None

        # Hip tracking for ball-behind detection
        self._hip_history: deque = deque(maxlen=15)  # 0.5 sec at 30fps
        self._current_hip_pos: Optional[Tuple[float, float]] = None

        # Direction history for fallback when player stops (STATIC direction)
        self._direction_history: deque = deque(maxlen=15)  # Track recent directions

        # Turning zones (loaded from JSON annotations)
        self._turning_zones: Optional[TurningZoneSet] = None
        self._figure8_layout: Optional[Figure8Layout] = None

        # Ball-behind detection config
        # NOTE: Must match BALL_POSITION_THRESHOLD in video/annotate_with_json_cones.py
        self._behind_threshold = 20.0  # Pixels - ball must be this far behind hip
        self._behind_sustained_frames = 15  # ~0.5 sec at 30fps to confirm loss
        self._movement_threshold = 3.0  # Min hip movement to determine direction

        logger.info(f"BallControlDetector initialized (video: {self._video_width}x{self._video_height})")

    def _load_video_dimensions(self, video_path: str) -> None:
        """
        Load video dimensions using ffprobe.

        Args:
            video_path: Path to video file
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height',
                    '-of', 'csv=p=0',
                    video_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    self._video_width = int(parts[0])
                    self._video_height = int(parts[1])
                    logger.info(f"Loaded video dimensions: {self._video_width}x{self._video_height}")
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not get video dimensions: {e}. Using default 1920x1080")

    def detect(
        self,
        ball_df: pd.DataFrame,
        pose_df: pd.DataFrame,
        cone_df: pd.DataFrame,
        fps: float = 30.0
    ) -> DetectionResult:
        """
        Run ball control detection for Figure-8 drill.

        Args:
            ball_df: Ball detection DataFrame
            pose_df: Pose keypoint DataFrame
            cone_df: Cone detection DataFrame
            fps: Video FPS for timestamps

        Returns:
            DetectionResult with events and frame data
        """
        try:
            logger.info("Starting Figure-8 drill detection...")
            logger.info(f"  Ball frames: {len(ball_df)}")
            logger.info(f"  Pose records: {len(pose_df)}")
            logger.info(f"  Cone records: {len(cone_df)}")

            # Log video dimensions and ball data range for debugging
            # NOTE: Do NOT infer video dimensions from ball data - ball can go off-screen
            # during boundary violations, which would incorrectly suggest larger dimensions.
            if 'center_x' in ball_df.columns:
                logger.info(f"  Video dimensions: {self._video_width}x{self._video_height}")
                logger.info(f"  Ball X range: {ball_df['center_x'].min():.0f} - {ball_df['center_x'].max():.0f}")

            # Reset state
            self._reset_state()

            # Initialize Figure-8 detector and identify cone roles
            # Passing parquet_dir enables loading manual cone annotations if available
            self._f8_detector = Figure8ConeDetector(self._drill_config, parquet_dir=self.parquet_dir)
            cone_roles = self._f8_detector.identify_cone_roles(cone_df, frame_id=0)
            self._f8_detector.setup_gates(cone_roles)

            # Store cone roles for use in _get_nearest_cone (static positions from JSON)
            self._cone_roles = cone_roles

            logger.info(f"Cone roles identified: {len(cone_roles)} cones")

            # Load Figure8Layout and create turning zones for ball-behind detection
            if self.parquet_dir:
                try:
                    json_path = Path(self.parquet_dir) / "cone_annotations.json"
                    if json_path.exists():
                        self._figure8_layout = load_cone_annotations(str(json_path))
                        zone_config = TurningZoneConfig.default()
                        self._turning_zones = create_turning_zones(self._figure8_layout, zone_config)
                        logger.info(f"Turning zones created: START and GATE2")
                except Exception as e:
                    logger.warning(f"Could not load turning zones: {e}")

            # Extract ankles and find closest per frame
            ankle_df = extract_ankle_positions(pose_df)
            merged_df = get_closest_ankle_per_frame(ankle_df, ball_df)

            if merged_df.empty:
                return DetectionResult(
                    success=False,
                    total_frames=0,
                    events=[],
                    frame_data=[],
                    error="No valid frames after merging"
                )

            # Merge with ball data (include interpolated flag for filtering)
            ball_cols = ['frame_id', 'center_x', 'center_y',
                        'field_center_x', 'field_center_y', 'interpolated']
            available_cols = [c for c in ball_cols if c in ball_df.columns]
            merged_df = merged_df.merge(ball_df[available_cols], on='frame_id')

            merged_df.rename(columns={
                'center_x': 'ball_x',
                'center_y': 'ball_y',
                'field_center_x': 'ball_field_x',
                'field_center_y': 'ball_field_y',
            }, inplace=True)

            # Calculate ball velocity (field coordinates - for general detection)
            merged_df = merged_df.sort_values('frame_id')
            merged_df['ball_velocity'] = np.sqrt(
                merged_df['ball_field_x'].diff()**2 +
                merged_df['ball_field_y'].diff()**2
            ).fillna(0)

            # Calculate pixel velocity (for boundary stuck detection)
            # Pixel velocity is needed because boundary thresholds are in pixel units
            merged_df['ball_velocity_pixel'] = np.sqrt(
                merged_df['ball_x'].diff()**2 +
                merged_df['ball_y'].diff()**2
            ).fillna(0)

            # Process each frame
            total_frames = len(merged_df)
            processed_frames = set()

            for _, row in merged_df.iterrows():
                frame_id = int(row['frame_id'])
                timestamp = frame_id / fps
                processed_frames.add(frame_id)

                frame_result = self._analyze_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    row=row,
                    pose_df=pose_df  # Pass pose data for hip extraction
                )

                if frame_result:
                    self._frame_data.append(frame_result)

            # Check ball-only frames for boundary violations
            # (handles cases where player is off-screen but ball is stuck at edge)
            self._check_ball_only_boundary_violations(ball_df, processed_frames, fps)

            # Finalize any open events
            self._finalize_events()

            result = DetectionResult(
                success=True,
                total_frames=total_frames,
                events=self._events,
                frame_data=self._frame_data,
                gate_passages=self._f8_detector.gate_passages if self._f8_detector else [],
                cone_roles=cone_roles,
                total_laps=self._f8_detector.lap_count if self._f8_detector else 0,
            )

            logger.info(f"Detection complete:")
            logger.info(f"  Processed frames: {total_frames}")
            logger.info(f"  Loss events: {result.total_loss_events}")
            logger.info(f"  Control percentage: {result.control_percentage:.1f}%")

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return DetectionResult(
                success=False,
                total_frames=0,
                events=[],
                frame_data=[],
                error=str(e)
            )

    def _reset_state(self):
        """Reset internal state for new detection run."""
        self._current_state = ControlState.CONTROLLED
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = DrillPhase.AT_START
        self._events = []
        self._frame_data = []
        self._event_counter = 0
        self._prev_ball_pos = None

        # Reset hip tracking
        self._hip_history.clear()
        self._current_hip_pos = None

        # Reset direction history
        self._direction_history.clear()

        if self._f8_detector:
            self._f8_detector.reset()

    def _analyze_frame(
        self,
        frame_id: int,
        timestamp: float,
        row: pd.Series,
        pose_df: Optional[pd.DataFrame] = None
    ) -> Optional[FrameData]:
        """
        Analyze a single frame.

        Gathers data and calls detect_loss() to determine control state.
        Uses static cone positions from JSON annotations (stored in self._cone_roles).
        """
        # Get positions
        ball_pos = (row['ball_field_x'], row['ball_field_y'])
        ball_pixel_pos = (row['ball_x'], row['ball_y'])
        ankle_pos = (row['ankle_field_x'], row['ankle_field_y'])
        distance = row['ball_ankle_distance']
        velocity = row['ball_velocity']
        velocity_pixel = row.get('ball_velocity_pixel', velocity)  # Pixel velocity for boundary detection

        # Extract hip position for ball-behind detection
        hip_pixel_pos: Optional[Tuple[float, float]] = None
        if pose_df is not None:
            hip_pixel_pos = self._extract_hip_position(frame_id, pose_df)

        # Update hip history and calculate player movement direction
        player_direction: Optional[str] = None
        if hip_pixel_pos is not None:
            self._hip_history.append(hip_pixel_pos)
            self._current_hip_pos = hip_pixel_pos

            # Calculate direction from hip movement
            if len(self._hip_history) >= 2:
                prev_hip = self._hip_history[0]
                dx = hip_pixel_pos[0] - prev_hip[0]
                if dx > self._movement_threshold:
                    player_direction = "RIGHT"  # Moving toward start cone
                elif dx < -self._movement_threshold:
                    player_direction = "LEFT"  # Moving toward gate 2

        # Store direction in history (including None for STATIC)
        self._direction_history.append(player_direction)

        # If direction is None (STATIC), fall back to last known direction
        # This prevents ball-behind detection from being skipped when player stops
        if player_direction is None:
            for d in reversed(self._direction_history):
                if d is not None:
                    player_direction = d
                    break

        # Check if ball is in turning zone
        in_turning_zone: Optional[str] = None
        if self._turning_zones is not None:
            in_turning_zone = self._turning_zones.get_zone_at_point(
                ball_pixel_pos[0], ball_pixel_pos[1]
            )

        # ============================================================
        # DETECTION LOGIC - calls detect_loss()
        # ============================================================
        is_loss, loss_type = self.detect_loss(
            ball_pos=ball_pos,
            ball_pixel_pos=ball_pixel_pos,
            ankle_pos=ankle_pos,
            distance=distance,
            velocity=velocity,
            velocity_pixel=velocity_pixel,
            frame_id=frame_id,
            timestamp=timestamp,
            history=self._frame_data,
            hip_pixel_pos=hip_pixel_pos,
            player_direction=player_direction,
            in_turning_zone=in_turning_zone
        )

        # Determine control state from detection result
        if is_loss:
            control_state = ControlState.LOST
        else:
            control_state = ControlState.CONTROLLED

        # Handle state transitions (create/close events)
        self._handle_state_change(
            new_state=control_state,
            loss_type=loss_type,
            frame_id=frame_id,
            timestamp=timestamp,
            ball_pos=ball_pos,
            ankle_pos=ankle_pos,
            distance=distance,
            velocity=velocity,
            row=row
        )

        # Get nearest cone (uses static positions from JSON annotations)
        # JSON cone positions are in pixel coords, so use ball pixel position
        ball_pixel_pos = (row['ball_x'], row['ball_y'])
        nearest_cone_id, nearest_cone_dist = self._get_nearest_cone(ball_pixel_pos)

        # Figure-8 tracking (gate passages, direction, phase)
        drill_phase = None
        drill_direction = None
        current_gate = None

        if self._f8_detector:
            if self._prev_ball_pos:
                dx = ball_pos[0] - self._prev_ball_pos[0]
                if abs(dx) > 5:
                    drill_direction = DrillDirection.FORWARD if dx > 0 else DrillDirection.BACKWARD
                else:
                    drill_direction = DrillDirection.STATIONARY
                self._current_direction = drill_direction

                # Detect gate passage
                passage = self._f8_detector.detect_gate_passage(
                    self._prev_ball_pos,
                    ball_pos,
                    frame_id,
                    timestamp,
                    ankle_pos,
                    control_state == ControlState.CONTROLLED
                )

                if passage:
                    current_gate = passage.gate_id
                    self._f8_detector.update_lap_count(passage)
            else:
                drill_direction = DrillDirection.STATIONARY

            drill_phase = self._f8_detector.get_current_phase(
                ball_pos, self._current_direction
            )
            self._current_phase = drill_phase

        self._prev_ball_pos = ball_pos

        # Simple control score (just for reporting, not used in detection)
        control_score = max(0.0, 1.0 - (distance / self._detection_config.loss_distance_threshold))

        # Calculate ball-behind status for this frame
        ball_behind = self._is_ball_behind(
            ball_pixel_pos, hip_pixel_pos, player_direction
        )

        return FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            ball_x=row['ball_x'],
            ball_y=row['ball_y'],
            ball_field_x=row['ball_field_x'],
            ball_field_y=row['ball_field_y'],
            ball_velocity=velocity,
            ankle_x=row['ankle_x'],
            ankle_y=row['ankle_y'],
            ankle_field_x=row['ankle_field_x'],
            ankle_field_y=row['ankle_field_y'],
            closest_ankle=row['closest_ankle'],
            nearest_cone_id=nearest_cone_id,
            nearest_cone_distance=nearest_cone_dist,
            current_gate=current_gate,
            ball_ankle_distance=distance,
            control_score=control_score,
            control_state=control_state,
            drill_phase=drill_phase,
            drill_direction=drill_direction,
            lap_count=self._f8_detector.lap_count if self._f8_detector else 0,
            # New fields for ball-behind detection
            hip_x=hip_pixel_pos[0] if hip_pixel_pos else None,
            hip_y=hip_pixel_pos[1] if hip_pixel_pos else None,
            player_movement_direction=player_direction,
            ball_behind_player=ball_behind,
            in_turning_zone=in_turning_zone,
        )

    # ================================================================
    # HELPER METHODS FOR BALL-BEHIND DETECTION
    # ================================================================

    def _extract_hip_position(
        self,
        frame_id: int,
        pose_df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """
        Extract hip position from pose data for a given frame.

        Args:
            frame_id: Frame number to extract hip for
            pose_df: Full pose DataFrame

        Returns:
            (hip_x, hip_y) in pixels, or None if not found
        """
        # Look for 'hip' keypoint in this frame
        frame_pose = pose_df[pose_df['frame_idx'] == frame_id]
        if frame_pose.empty:
            return None

        # Find hip keypoint (try 'hip' first, then calculate from left/right hip)
        hip_row = frame_pose[frame_pose['keypoint_name'] == 'hip']
        if not hip_row.empty:
            hip = hip_row.iloc[0]
            if hip['confidence'] >= 0.3:  # Minimum confidence threshold
                return (float(hip['x']), float(hip['y']))

        # Fallback: calculate from left_hip and right_hip
        left_hip = frame_pose[frame_pose['keypoint_name'] == 'left_hip']
        right_hip = frame_pose[frame_pose['keypoint_name'] == 'right_hip']

        if not left_hip.empty and not right_hip.empty:
            lh = left_hip.iloc[0]
            rh = right_hip.iloc[0]
            if lh['confidence'] >= 0.3 and rh['confidence'] >= 0.3:
                hip_x = (lh['x'] + rh['x']) / 2
                hip_y = (lh['y'] + rh['y']) / 2
                return (float(hip_x), float(hip_y))

        return None

    def _is_ball_behind(
        self,
        ball_pixel_pos: Tuple[float, float],
        hip_pixel_pos: Optional[Tuple[float, float]],
        player_direction: Optional[str]
    ) -> Optional[bool]:
        """
        Determine if ball is behind player relative to movement direction.

        Args:
            ball_pixel_pos: Ball position in pixels
            hip_pixel_pos: Hip position in pixels (or None)
            player_direction: "LEFT", "RIGHT", or None

        Returns:
            True if ball is behind, False if in front, None if can't determine
        """
        if hip_pixel_pos is None or player_direction is None:
            return None

        delta_x = ball_pixel_pos[0] - hip_pixel_pos[0]

        # Ball is "behind" if it's opposite to movement direction
        if player_direction == "LEFT":
            # Moving left = forward direction, ball to RIGHT of hip = behind
            return delta_x > self._behind_threshold
        elif player_direction == "RIGHT":
            # Moving right = backward direction, ball to LEFT of hip = behind
            return delta_x < -self._behind_threshold

        return None

    # ================================================================
    # DETECTION LOGIC - CATEGORIZED LOSS DETECTION
    # ================================================================
    def detect_loss(
        self,
        ball_pos: Tuple[float, float],
        ball_pixel_pos: Tuple[float, float],
        ankle_pos: Tuple[float, float],
        distance: float,
        velocity: float,
        frame_id: int,
        timestamp: float,
        history: List[FrameData],
        hip_pixel_pos: Optional[Tuple[float, float]] = None,
        player_direction: Optional[str] = None,
        in_turning_zone: Optional[str] = None,
        velocity_pixel: Optional[float] = None
    ) -> Tuple[bool, Optional[EventType]]:
        """
        Detect if ball control is lost.

        Detects two types of loss events:
        1. BOUNDARY_VIOLATION - Ball exits video frame (out of bounds)
        2. BALL_BEHIND_PLAYER - Ball stays behind player for sustained period

        Args:
            ball_pos: Ball position (field_x, field_y)
            ball_pixel_pos: Ball position in pixels (x, y)
            ankle_pos: Closest ankle position (field_x, field_y)
            distance: Ball-to-ankle distance (pre-calculated)
            velocity: Ball velocity this frame (field coordinates)
            frame_id: Current frame number
            timestamp: Current timestamp in seconds
            history: List of previous FrameData (for temporal analysis)
            hip_pixel_pos: Player hip position in pixels (for ball-behind detection)
            player_direction: "LEFT", "RIGHT", or None
            in_turning_zone: "START", "GATE2", or None (suppress ball-behind in zones)
            velocity_pixel: Ball velocity in pixels/frame (for boundary stuck detection)

        Returns:
            Tuple of:
            - is_loss: True if control is lost, False otherwise
            - loss_type: EventType if loss detected, None otherwise
        """
        # ============================================================
        # 1. BOUNDARY VIOLATION - Ball exits video frame
        # ============================================================
        # Priority: This is checked first as it's the most severe loss type.

        EDGE_MARGIN = 50
        SCREEN_RIGHT_THRESHOLD = self._video_width - EDGE_MARGIN
        SCREEN_LEFT_THRESHOLD = EDGE_MARGIN
        APPROACHING_RIGHT_THRESHOLD = self._video_width - 100
        APPROACHING_LEFT_THRESHOLD = 100
        # Use pixel velocity for stuck detection (5 px/frame threshold)
        # Previous threshold of 0.5 was for field coordinates which are much smaller
        STUCK_VELOCITY_THRESHOLD_PIXEL = 5.0
        MIN_TIMESTAMP = 3.0
        FRAMES_TO_CHECK = 5

        # Use pixel velocity if available, otherwise fall back to field velocity
        # (with a scaled threshold for backward compatibility)
        effective_velocity = velocity_pixel if velocity_pixel is not None else velocity
        effective_threshold = STUCK_VELOCITY_THRESHOLD_PIXEL if velocity_pixel is not None else 0.5

        # Skip early frames
        if timestamp < MIN_TIMESTAMP:
            return False, None

        ball_x_pixel = ball_pixel_pos[0]

        # Check RIGHT edge
        if ball_x_pixel > SCREEN_RIGHT_THRESHOLD and effective_velocity < effective_threshold:
            if len(history) >= FRAMES_TO_CHECK:
                recent = history[-FRAMES_TO_CHECK:]
                x_deltas = [recent[i].ball_x - recent[i-1].ball_x for i in range(1, len(recent))]
                if x_deltas:
                    avg_x_delta = sum(x_deltas) / len(x_deltas)
                    if avg_x_delta >= 0:
                        all_at_edge = all(f.ball_x > APPROACHING_RIGHT_THRESHOLD for f in recent)
                        if all_at_edge:
                            logger.debug(
                                f"Frame {frame_id}: BOUNDARY_VIOLATION at RIGHT edge "
                                f"(x={ball_x_pixel:.0f}px, velocity={effective_velocity:.1f}px/f)"
                            )
                            return True, EventType.BOUNDARY_VIOLATION

        # Check LEFT edge
        if ball_x_pixel < SCREEN_LEFT_THRESHOLD and effective_velocity < effective_threshold:
            if len(history) >= FRAMES_TO_CHECK:
                recent = history[-FRAMES_TO_CHECK:]
                x_deltas = [recent[i].ball_x - recent[i-1].ball_x for i in range(1, len(recent))]
                if x_deltas:
                    avg_x_delta = sum(x_deltas) / len(x_deltas)
                    if avg_x_delta <= 0:
                        all_at_edge = all(f.ball_x < APPROACHING_LEFT_THRESHOLD for f in recent)
                        if all_at_edge:
                            logger.debug(
                                f"Frame {frame_id}: BOUNDARY_VIOLATION at LEFT edge "
                                f"(x={ball_x_pixel:.0f}px, velocity={effective_velocity:.1f}px/f)"
                            )
                            return True, EventType.BOUNDARY_VIOLATION

        # ============================================================
        # 2. BALL_BEHIND_PLAYER - Ball stays behind for sustained period
        # ============================================================
        # Only trigger if:
        # - Player has clear movement direction
        # - Ball is behind player (opposite to movement direction)
        # - NOT in a turning zone (where "behind" is expected briefly)
        # - Sustained for N consecutive frames

        # Skip if in turning zone (turning zones are where "behind" is expected)
        if in_turning_zone is not None:
            return False, None

        # Skip if no direction information
        if hip_pixel_pos is None or player_direction is None:
            return False, None

        # Check if ball is currently behind player
        is_behind = self._is_ball_behind(ball_pixel_pos, hip_pixel_pos, player_direction)

        # If ball is NOT behind, check if we should continue LOST state
        # (require ball to be close enough before recovering)
        if not is_behind:
            # Check if we're currently in a LOST state (from recent history)
            if len(history) >= 5:
                recent_lost = sum(1 for f in history[-5:] if f.control_state == ControlState.LOST)
                if recent_lost >= 3:  # Majority of recent frames were LOST
                    # Calculate ball-hip distance to check recovery
                    ball_hip_dist = abs(ball_pixel_pos[0] - hip_pixel_pos[0])
                    # Require ball to be close enough to hip to recover
                    RECOVERY_DISTANCE_THRESHOLD = 80.0  # pixels
                    if ball_hip_dist > RECOVERY_DISTANCE_THRESHOLD:
                        # Ball still too far - continue LOST state
                        logger.debug(
                            f"Frame {frame_id}: Continuing BALL_BEHIND loss "
                            f"(ball-hip dist={ball_hip_dist:.0f}px, awaiting recovery)"
                        )
                        return True, EventType.BALL_BEHIND_PLAYER
            return False, None

        # Check for sustained "behind" pattern in history
        if len(history) >= self._behind_sustained_frames:
            recent = history[-self._behind_sustained_frames:]

            # Count consecutive frames where ball was behind
            behind_count = 0
            for frame in recent:
                # Check if this frame had ball behind (using stored value)
                if frame.ball_behind_player is True:
                    behind_count += 1
                else:
                    # Reset counter if we find a "not behind" frame
                    behind_count = 0

            # Only trigger if ball was behind for entire sustained period
            if behind_count >= self._behind_sustained_frames - 1:
                # Also verify player was moving consistently (not turning)
                directions = [f.player_movement_direction for f in recent if f.player_movement_direction]
                if len(directions) >= self._behind_sustained_frames // 2:
                    # Check if mostly same direction
                    dominant_direction = max(set(directions), key=directions.count)
                    same_direction_ratio = directions.count(dominant_direction) / len(directions)

                    if same_direction_ratio >= 0.7:  # 70% consistency threshold
                        logger.debug(
                            f"Frame {frame_id}: BALL_BEHIND_PLAYER detected "
                            f"(behind for {behind_count} frames, direction={dominant_direction})"
                        )
                        return True, EventType.BALL_BEHIND_PLAYER

        return False, None

    # ================================================================
    # EVENT LIFECYCLE (keep as-is)
    # ================================================================
    def _handle_state_change(
        self,
        new_state: ControlState,
        loss_type: Optional[EventType],
        frame_id: int,
        timestamp: float,
        ball_pos: Tuple[float, float],
        ankle_pos: Tuple[float, float],
        distance: float,
        velocity: float,
        row: pd.Series
    ):
        """Handle state transitions and create/close events."""
        prev_state = self._current_state

        # Transition to LOST - create new event
        if new_state == ControlState.LOST and prev_state != ControlState.LOST:
            event = LossEvent(
                event_id=self._event_counter,
                event_type=loss_type or EventType.LOSS_DISTANCE,
                start_frame=frame_id,
                end_frame=None,
                start_timestamp=timestamp,
                end_timestamp=None,
                ball_position=ball_pos,
                player_position=ankle_pos,
                distance_at_loss=distance,
                velocity_at_loss=velocity,
                nearest_cone_id=int(row.get('nearest_cone_id', -1)) if 'nearest_cone_id' in row else -1,
                gate_context=None,
                severity=self._get_severity(),
            )
            self._events.append(event)
            self._event_counter += 1
            logger.debug(f"Loss event started at frame {frame_id}")

        # Transition from LOST - close event
        elif prev_state == ControlState.LOST and new_state != ControlState.LOST:
            for event in reversed(self._events):
                if event.end_frame is None:
                    event.end_frame = frame_id
                    event.end_timestamp = timestamp
                    event.recovered = True
                    event.recovery_frame = frame_id
                    logger.debug(f"Loss event ended at frame {frame_id}")
                    break

        self._current_state = new_state

    def _get_severity(self) -> str:
        """Get severity based on current drill phase."""
        if self._current_phase in [DrillPhase.PASSING_G1, DrillPhase.PASSING_G2]:
            return "high"
        elif self._current_phase == DrillPhase.AT_TURN:
            return "high"
        elif self._current_phase == DrillPhase.BETWEEN_GATES:
            return "medium"
        return "low"

    def _check_ball_only_boundary_violations(
        self,
        ball_df: pd.DataFrame,
        processed_frames: set,
        fps: float
    ):
        """
        Check for boundary violations in frames that have ball data but no ankle data.

        This handles cases where the player goes off-screen chasing the ball,
        causing pose detection to fail, but the ball is still visible and stuck at edge.

        Args:
            ball_df: Ball detection DataFrame with center_x, center_y
            processed_frames: Set of frame IDs already processed in main loop
            fps: Video FPS for timestamp calculation
        """
        EDGE_MARGIN = 50
        SCREEN_RIGHT_THRESHOLD = self._video_width - EDGE_MARGIN
        SCREEN_LEFT_THRESHOLD = EDGE_MARGIN
        STUCK_VELOCITY_THRESHOLD = 5.0
        MIN_CONSECUTIVE_FRAMES = 5
        MIN_TIMESTAMP = 3.0

        # Get ball-only frames (frames with ball data but no ankle data)
        all_ball_frames = set(ball_df['frame_id'].unique())
        ball_only_frames = sorted(all_ball_frames - processed_frames)

        if not ball_only_frames:
            return

        logger.debug(f"Checking {len(ball_only_frames)} ball-only frames for boundary violations")

        # Calculate velocity on FULL ball data first (to avoid gaps in diff)
        ball_df_sorted = ball_df.sort_values('frame_id').copy()
        ball_df_sorted['velocity_pixel'] = np.sqrt(
            ball_df_sorted['center_x'].diff()**2 +
            ball_df_sorted['center_y'].diff()**2
        ).fillna(0)

        # Now filter to ball-only frames
        ball_subset = ball_df_sorted[ball_df_sorted['frame_id'].isin(ball_only_frames)]

        # Track consecutive edge frames
        edge_start = None
        edge_type = None  # "LEFT" or "RIGHT"
        consecutive_count = 0

        for _, row in ball_subset.iterrows():
            frame_id = int(row['frame_id'])
            timestamp = frame_id / fps
            ball_x = row['center_x']
            velocity = row['velocity_pixel']

            # Skip early frames
            if timestamp < MIN_TIMESTAMP:
                continue

            # Check if ball is at edge and stuck
            at_right_edge = ball_x > SCREEN_RIGHT_THRESHOLD and velocity < STUCK_VELOCITY_THRESHOLD
            at_left_edge = ball_x < SCREEN_LEFT_THRESHOLD and velocity < STUCK_VELOCITY_THRESHOLD

            current_edge = "RIGHT" if at_right_edge else ("LEFT" if at_left_edge else None)

            if current_edge:
                if edge_start is None:
                    edge_start = frame_id
                    edge_type = current_edge
                    consecutive_count = 1
                elif current_edge == edge_type:
                    consecutive_count += 1
                else:
                    # Edge type changed, check if previous sequence was long enough
                    if consecutive_count >= MIN_CONSECUTIVE_FRAMES:
                        self._create_ball_only_boundary_event(
                            edge_start, frame_id - 1, edge_type, fps
                        )
                    edge_start = frame_id
                    edge_type = current_edge
                    consecutive_count = 1
            else:
                # Ball moved away from edge
                if edge_start is not None and consecutive_count >= MIN_CONSECUTIVE_FRAMES:
                    self._create_ball_only_boundary_event(
                        edge_start, frame_id - 1, edge_type, fps
                    )
                edge_start = None
                edge_type = None
                consecutive_count = 0

        # Handle sequence that extends to end of ball-only frames
        if edge_start is not None and consecutive_count >= MIN_CONSECUTIVE_FRAMES:
            self._create_ball_only_boundary_event(
                edge_start, int(ball_subset['frame_id'].iloc[-1]), edge_type, fps
            )

    def _create_ball_only_boundary_event(
        self,
        start_frame: int,
        end_frame: int,
        edge_type: str,
        fps: float
    ):
        """Create a boundary violation event for ball-only frames."""
        self._event_counter += 1

        # Placeholder positions for ball-only events (player is off-screen)
        edge_x = self._video_width if edge_type == "RIGHT" else 0.0
        placeholder_pos = (edge_x, self._video_height / 2)

        event = LossEvent(
            event_id=self._event_counter,
            event_type=EventType.BOUNDARY_VIOLATION,
            start_frame=start_frame,
            end_frame=end_frame,
            start_timestamp=start_frame / fps,
            end_timestamp=end_frame / fps,
            ball_position=placeholder_pos,
            player_position=placeholder_pos,  # Unknown - player off-screen
            distance_at_loss=0.0,  # Unknown - no ankle data
            velocity_at_loss=0.0,  # Ball is stuck at edge
            nearest_cone_id=-1,  # Not relevant for boundary violations
            gate_context=None,
            severity="high",  # Boundary violations are always high severity
            notes=f"Ball stuck at {edge_type} edge (player off-screen)"
        )

        self._events.append(event)
        logger.info(
            f"Ball-only boundary violation: frames {start_frame}-{end_frame} "
            f"({event.start_timestamp:.2f}s-{event.end_timestamp:.2f}s) at {edge_type} edge"
        )

    def _finalize_events(self):
        """Close any open events, merge overlapping, and filter short events."""
        MIN_EVENT_FRAMES = 15  # Minimum 0.5 seconds at 30fps

        for event in self._events:
            if event.end_frame is None:
                event.notes += " [Unclosed at end of video]"

        # Merge overlapping events of the same type
        self._events = self._merge_overlapping_events(self._events)

        # Filter out very short events (likely false positives)
        original_count = len(self._events)
        self._events = [e for e in self._events if e.duration_frames >= MIN_EVENT_FRAMES]
        filtered_count = original_count - len(self._events)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} short events (<{MIN_EVENT_FRAMES} frames)")

    def _merge_overlapping_events(self, events: List[LossEvent]) -> List[LossEvent]:
        """
        Merge events that overlap in time and are of the same type.

        This handles cases where ball-only boundary detection creates events
        that overlap with regular boundary detection events.
        """
        if len(events) <= 1:
            return events

        # Sort events by start frame
        sorted_events = sorted(events, key=lambda e: e.start_frame)
        merged = []

        for event in sorted_events:
            if not merged:
                merged.append(event)
                continue

            last = merged[-1]

            # Check if events overlap or are adjacent (within 5 frames)
            # and are the same type
            overlap_threshold = 5  # frames
            events_overlap = (
                event.event_type == last.event_type and
                event.start_frame <= (last.end_frame or last.start_frame) + overlap_threshold
            )

            if events_overlap:
                # Merge: extend last event's end frame
                new_end = max(last.end_frame or last.start_frame, event.end_frame or event.start_frame)
                last.end_frame = new_end
                last.end_timestamp = new_end / 30.0  # Assume 30fps
                # Combine notes if different
                if event.notes and event.notes not in (last.notes or ""):
                    last.notes = f"{last.notes or ''} {event.notes}".strip()
                logger.debug(
                    f"Merged overlapping {event.event_type.name} events: "
                    f"{last.start_frame}-{last.end_frame}"
                )
            else:
                merged.append(event)

        if len(merged) < len(events):
            logger.info(f"Merged {len(events) - len(merged)} overlapping events")

        return merged

    def _get_nearest_cone(
        self,
        ball_pos: Tuple[float, float]
    ) -> Tuple[int, float]:
        """
        Get nearest cone to ball position using static JSON annotations.

        Uses self._cone_roles which contains static cone positions from
        cone_annotations.json. No longer depends on per-frame cone_df lookups.
        """
        if not self._cone_roles:
            return -1, float('inf')

        min_dist = float('inf')
        nearest_id = -1

        for cone in self._cone_roles:
            # field_x and field_y are actually pixel coordinates from JSON (px, py)
            dist = np.sqrt(
                (cone.field_x - ball_pos[0])**2 +
                (cone.field_y - ball_pos[1])**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_id = cone.cone_id

        return nearest_id, min_dist


# Convenience function
def detect_ball_control(
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    cone_df: pd.DataFrame,
    config: Optional[AppConfig] = None,
    fps: float = 30.0,
    parquet_dir: Optional[str] = None,
    video_path: Optional[str] = None
) -> DetectionResult:
    """
    Convenience function for Figure-8 drill detection.

    Args:
        ball_df: Ball detection DataFrame
        pose_df: Pose keypoint DataFrame
        cone_df: Cone detection DataFrame
        config: Optional AppConfig (defaults to Figure-8 config)
        fps: Video FPS
        parquet_dir: Path to parquet directory for loading manual cone annotations
        video_path: Path to video file for getting actual video dimensions

    Returns:
        DetectionResult with Figure-8 specific metrics
    """
    if config is None:
        config = AppConfig.for_figure8()
    detector = BallControlDetector(config, parquet_dir=parquet_dir, video_path=video_path)
    return detector.detect(ball_df, pose_df, cone_df, fps)
