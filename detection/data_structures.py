"""
Data structures for Ball Control Detection System.

Defines all data models for detection results including:
- ControlState: Ball control state machine
- EventType: Types of loss events
- FrameData: Per-frame analysis data
- LossEvent: A detected loss-of-control event
- DetectionResult: Complete detection output

Figure-8 specific structures:
- DrillPhase: Current phase in Figure-8 drill
- GatePassage: Record of passing through a gate
- DrillDirection: Direction of travel (forward/backward)
- ConeAnnotation: Single cone from JSON annotation
- Figure8Layout: Complete drill layout with gate definitions
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# CONE ANNOTATION STRUCTURES (from JSON)
# =============================================================================

@dataclass
class ConeAnnotation:
    """
    Single cone from JSON annotation file.

    Attributes:
        role: Cone role ("start", "gate1_left", "gate1_right", "gate2_left", "gate2_right")
        px: Pixel x-coordinate (horizontal position in video frame)
        py: Pixel y-coordinate (vertical position in video frame)
        bbox: Optional bounding box dict with x1, y1, x2, y2
    """
    role: str
    px: float
    py: float
    bbox: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'role': self.role,
            'px': self.px,
            'py': self.py,
            'bbox': self.bbox,
        }

    @classmethod
    def from_json(cls, role: str, data: dict) -> 'ConeAnnotation':
        """Create from JSON annotation data."""
        return cls(
            role=role,
            px=data['px'],
            py=data['py'],
            bbox=data.get('bbox'),
        )


@dataclass
class Figure8Layout:
    """
    Complete Figure-8 drill layout from JSON annotations.

    Provides gate line definitions for passage detection using pixel coordinates.

    Layout (pixel x increases left to right):
    [G2_LEFT] [G2_RIGHT] ---- [G1_LEFT] [G1_RIGHT] ---- [START]

    Forward direction: Start → G1 → G2 (decreasing px)
    Backward direction: G2 → G1 → Start (increasing px)
    """
    start_cone: ConeAnnotation
    gate1_left: ConeAnnotation
    gate1_right: ConeAnnotation
    gate2_left: ConeAnnotation
    gate2_right: ConeAnnotation
    video_name: Optional[str] = None
    annotated_at: Optional[str] = None

    @property
    def gate1_line(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Gate 1 line segment as ((x1, y1), (x2, y2)) in pixel coords."""
        return (
            (self.gate1_left.px, self.gate1_left.py),
            (self.gate1_right.px, self.gate1_right.py)
        )

    @property
    def gate2_line(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Gate 2 line segment as ((x1, y1), (x2, y2)) in pixel coords."""
        return (
            (self.gate2_left.px, self.gate2_left.py),
            (self.gate2_right.px, self.gate2_right.py)
        )

    @property
    def gate1_center(self) -> Tuple[float, float]:
        """Center point of Gate 1 in pixel coords."""
        return (
            (self.gate1_left.px + self.gate1_right.px) / 2,
            (self.gate1_left.py + self.gate1_right.py) / 2
        )

    @property
    def gate2_center(self) -> Tuple[float, float]:
        """Center point of Gate 2 in pixel coords."""
        return (
            (self.gate2_left.px + self.gate2_right.px) / 2,
            (self.gate2_left.py + self.gate2_right.py) / 2
        )

    @property
    def start_position(self) -> Tuple[float, float]:
        """Start cone position in pixel coords."""
        return (self.start_cone.px, self.start_cone.py)

    @property
    def gate1_width(self) -> float:
        """Width of Gate 1 in pixels."""
        import math
        return math.sqrt(
            (self.gate1_right.px - self.gate1_left.px)**2 +
            (self.gate1_right.py - self.gate1_left.py)**2
        )

    @property
    def gate2_width(self) -> float:
        """Width of Gate 2 in pixels."""
        import math
        return math.sqrt(
            (self.gate2_right.px - self.gate2_left.px)**2 +
            (self.gate2_right.py - self.gate2_left.py)**2
        )

    def get_all_cones(self) -> List[ConeAnnotation]:
        """Get all cones as a list."""
        return [
            self.start_cone,
            self.gate1_left,
            self.gate1_right,
            self.gate2_left,
            self.gate2_right,
        ]

    def to_cone_roles(self) -> List['ConeRole']:
        """Convert to list of ConeRole objects for export."""
        roles = []
        for cone in self.get_all_cones():
            roles.append(ConeRole(
                cone_id=-1,  # No ID from JSON, use -1
                role=cone.role,
                field_x=cone.px,  # Using pixel coords
                field_y=cone.py,
            ))
        return roles

    @classmethod
    def from_json(cls, data: dict) -> 'Figure8Layout':
        """Create from parsed JSON annotation file."""
        cones = data['cones']
        return cls(
            start_cone=ConeAnnotation.from_json('start', cones['start']),
            gate1_left=ConeAnnotation.from_json('gate1_left', cones['gate1_left']),
            gate1_right=ConeAnnotation.from_json('gate1_right', cones['gate1_right']),
            gate2_left=ConeAnnotation.from_json('gate2_left', cones['gate2_left']),
            gate2_right=ConeAnnotation.from_json('gate2_right', cones['gate2_right']),
            video_name=data.get('video'),
            annotated_at=data.get('annotated_at'),
        )


# =============================================================================
# DETECTION STATE ENUMS
# =============================================================================


class ControlState(Enum):
    """Ball control states."""
    CONTROLLED = "controlled"
    TRANSITION = "transition"
    LOST = "lost"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Loss-of-control event types."""
    LOSS_DISTANCE = "loss_distance"
    LOSS_VELOCITY = "loss_velocity"
    LOSS_DIRECTION = "loss_direction"
    RECOVERY = "recovery"
    BOUNDARY_VIOLATION = "boundary"
    BALL_BEHIND_PLAYER = "ball_behind"  # Ball stays behind player relative to movement


class DrillPhase(Enum):
    """
    Current phase in Figure-8 drill.

    Drill pattern: START → G1 → G2 → TURN → G2 → G1 → (repeat)
    """
    AT_START = "at_start"              # Near start cone, ready to begin
    APPROACHING_G1 = "approaching_g1"  # Moving toward Gate 1
    PASSING_G1 = "passing_g1"          # Currently passing through Gate 1
    BETWEEN_GATES = "between_gates"    # Between G1 and G2
    APPROACHING_G2 = "approaching_g2"  # Moving toward Gate 2
    PASSING_G2 = "passing_g2"          # Currently passing through Gate 2
    AT_TURN = "at_turn"                # Beyond G2, turning around
    RETURNING = "returning"            # On return journey
    COMPLETED = "completed"            # Drill finished


class DrillDirection(Enum):
    """Direction of travel in drill."""
    FORWARD = "forward"    # Start → G2 direction
    BACKWARD = "backward"  # G2 → Start direction
    STATIONARY = "stationary"


@dataclass
class GatePassage:
    """Record of a gate passage event."""
    gate_id: str           # "G1" or "G2"
    direction: DrillDirection
    frame_id: int
    timestamp: float
    ball_position: Tuple[float, float]
    player_position: Tuple[float, float]
    ball_controlled: bool  # Was ball under control during passage
    passage_quality: float # 0-1 score for how cleanly they passed through

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gate_id': self.gate_id,
            'direction': self.direction.value,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'ball_x': self.ball_position[0],
            'ball_y': self.ball_position[1],
            'player_x': self.player_position[0],
            'player_y': self.player_position[1],
            'ball_controlled': self.ball_controlled,
            'passage_quality': self.passage_quality,
        }


@dataclass
class ConeRole:
    """Cone role assignment in Figure-8 drill."""
    cone_id: int
    role: str  # "start", "gate1_left", "gate1_right", "gate2_left", "gate2_right"
    field_x: float
    field_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cone_id': self.cone_id,
            'role': self.role,
            'field_x': self.field_x,
            'field_y': self.field_y,
        }


@dataclass
class FrameData:
    """Data for a single frame analysis."""
    frame_id: int
    timestamp: float

    # Ball position
    ball_x: float
    ball_y: float
    ball_field_x: float
    ball_field_y: float
    ball_velocity: float

    # Player ankle position (closest)
    ankle_x: float
    ankle_y: float
    ankle_field_x: float
    ankle_field_y: float
    closest_ankle: str  # "left_ankle" or "right_ankle"

    # Context
    nearest_cone_id: int
    nearest_cone_distance: float
    current_gate: Optional[str]

    # Computed metrics
    ball_ankle_distance: float
    control_score: float
    control_state: ControlState

    # Figure-8 specific fields (optional, None for 7-cone drill)
    drill_phase: Optional[DrillPhase] = None
    drill_direction: Optional[DrillDirection] = None
    lap_count: int = 0  # Number of completed laps

    # Hip position (for ball-behind detection)
    hip_x: Optional[float] = None  # Hip pixel X coordinate
    hip_y: Optional[float] = None  # Hip pixel Y coordinate

    # Ball position relative to player (for ball-behind detection)
    player_movement_direction: Optional[str] = None  # "LEFT", "RIGHT", or None
    ball_behind_player: Optional[bool] = None  # True if ball is behind player
    in_turning_zone: Optional[str] = None  # "START", "GATE2", or None

    # Ball tracking quality (for filtering false positives)
    ball_interpolated: bool = False  # True if ball position is interpolated (not real detection)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        result = {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'ball_x': self.ball_x,
            'ball_y': self.ball_y,
            'ball_field_x': self.ball_field_x,
            'ball_field_y': self.ball_field_y,
            'ball_velocity': self.ball_velocity,
            'ankle_x': self.ankle_x,
            'ankle_y': self.ankle_y,
            'ankle_field_x': self.ankle_field_x,
            'ankle_field_y': self.ankle_field_y,
            'closest_ankle': self.closest_ankle,
            'nearest_cone_id': self.nearest_cone_id,
            'nearest_cone_distance': self.nearest_cone_distance,
            'current_gate': self.current_gate,
            'ball_ankle_distance': self.ball_ankle_distance,
            'control_score': self.control_score,
            'control_state': self.control_state.value,
        }
        # Add Figure-8 specific fields if present
        if self.drill_phase is not None:
            result['drill_phase'] = self.drill_phase.value
        if self.drill_direction is not None:
            result['drill_direction'] = self.drill_direction.value
        result['lap_count'] = self.lap_count

        # Add hip/ball-behind fields if present
        if self.hip_x is not None:
            result['hip_x'] = self.hip_x
        if self.hip_y is not None:
            result['hip_y'] = self.hip_y
        if self.player_movement_direction is not None:
            result['player_movement_direction'] = self.player_movement_direction
        if self.ball_behind_player is not None:
            result['ball_behind_player'] = self.ball_behind_player
        if self.in_turning_zone is not None:
            result['in_turning_zone'] = self.in_turning_zone

        return result


@dataclass
class LossEvent:
    """A ball control loss event."""
    event_id: int
    event_type: EventType
    start_frame: int
    end_frame: Optional[int]
    start_timestamp: float
    end_timestamp: Optional[float]

    # Position at loss
    ball_position: Tuple[float, float]
    player_position: Tuple[float, float]
    distance_at_loss: float
    velocity_at_loss: float

    # Context
    nearest_cone_id: int
    gate_context: Optional[str]

    # Recovery
    recovered: bool = False
    recovery_frame: Optional[int] = None
    severity: str = "medium"
    notes: str = ""

    @property
    def duration_frames(self) -> int:
        if self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame

    @property
    def duration_seconds(self) -> float:
        if self.end_timestamp is None:
            return 0.0
        return self.end_timestamp - self.start_timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'duration_frames': self.duration_frames,
            'duration_seconds': self.duration_seconds,
            'ball_x': self.ball_position[0],
            'ball_y': self.ball_position[1],
            'player_x': self.player_position[0],
            'player_y': self.player_position[1],
            'distance_at_loss': self.distance_at_loss,
            'velocity_at_loss': self.velocity_at_loss,
            'nearest_cone_id': self.nearest_cone_id,
            'gate_context': self.gate_context,
            'recovered': self.recovered,
            'recovery_frame': self.recovery_frame,
            'severity': self.severity,
            'notes': self.notes,
        }


@dataclass
class DetectionResult:
    """Complete result from ball control detection."""
    success: bool
    total_frames: int
    events: List[LossEvent]
    frame_data: List[FrameData]

    # Summary
    total_loss_events: int = 0
    total_loss_duration_frames: int = 0
    control_percentage: float = 0.0

    error: Optional[str] = None

    # Figure-8 specific results (optional)
    gate_passages: List[GatePassage] = field(default_factory=list)
    cone_roles: List[ConeRole] = field(default_factory=list)
    total_laps: int = 0
    successful_passages: int = 0
    failed_passages: int = 0  # Passages where ball was not controlled

    def __post_init__(self):
        """Calculate summary statistics."""
        self.total_loss_events = len(self.events)
        self.total_loss_duration_frames = sum(
            e.duration_frames for e in self.events if e.end_frame
        )
        if self.total_frames > 0:
            controlled = self.total_frames - self.total_loss_duration_frames
            self.control_percentage = (controlled / self.total_frames) * 100

        # Calculate Figure-8 specific stats
        if self.gate_passages:
            self.successful_passages = sum(1 for p in self.gate_passages if p.ball_controlled)
            self.failed_passages = len(self.gate_passages) - self.successful_passages
