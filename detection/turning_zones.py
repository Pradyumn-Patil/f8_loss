"""
Turning Zones Module for Figure-8 Drill Analysis.

Defines elliptical turning zones around the START cone and GATE2 area
where players turn during Figure-8 drills. Zones are elliptical (not circular)
to compensate for camera perspective distortion.

Key components:
- TurningZone: Ellipse geometry with point-in-zone detection
- TurningZoneConfig: Configuration for zone sizes and stretch factors
- TurningZoneSet: Container for both zones with convenience methods
- create_turning_zones(): Factory to create zones from Figure8Layout
- draw_turning_zone()/draw_turning_zones(): Video visualization

Usage:
    from f8_loss.detection.turning_zones import create_turning_zones

    layout = load_cone_annotations(parquet_dir)
    zones = create_turning_zones(layout)

    if zones.is_in_turning_zone(ball_x, ball_y):
        print(f"Ball in: {zones.get_zone_at_point(ball_x, ball_y)}")
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data_structures import Figure8Layout

# Try to import OpenCV for drawing functions
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TurningZone:
    """
    Elliptical turning zone for Figure-8 drill.

    The ellipse is defined by center point, semi-axes, and optional rotation.
    Camera perspective distortion is handled via stretch factors applied
    during zone creation (semi_major vs semi_minor).

    Attributes:
        name: Zone identifier ("START" or "GATE2")
        center_px: X-coordinate of ellipse center (pixels)
        center_py: Y-coordinate of ellipse center (pixels)
        semi_major: Semi-major axis length (pixels) - typically horizontal
        semi_minor: Semi-minor axis length (pixels) - typically vertical
        rotation_deg: Rotation angle in degrees (0 = axes aligned with frame)
    """
    name: str
    center_px: float
    center_py: float
    semi_major: float
    semi_minor: float
    rotation_deg: float = 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the ellipse.

        Uses standard ellipse equation with rotation:
        For point (x', y') translated to ellipse center and rotated:
        (x'/a)² + (y'/b)² <= 1

        Args:
            x: X-coordinate to test (pixels)
            y: Y-coordinate to test (pixels)

        Returns:
            True if point is inside or on the ellipse boundary
        """
        # Translate point to ellipse-centered coordinates
        dx = x - self.center_px
        dy = y - self.center_py

        # Apply rotation (rotate point by negative angle to align with ellipse axes)
        theta = math.radians(self.rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Rotated coordinates
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        # Check ellipse equation: (x/a)² + (y/b)² <= 1
        # semi_major = a (horizontal axis after rotation)
        # semi_minor = b (vertical axis after rotation)
        if self.semi_major == 0 or self.semi_minor == 0:
            return False

        normalized = (x_rot / self.semi_major) ** 2 + (y_rot / self.semi_minor) ** 2
        return normalized <= 1.0

    def distance_to_center(self, x: float, y: float) -> float:
        """Calculate distance from point to zone center."""
        return math.sqrt((x - self.center_px) ** 2 + (y - self.center_py) ** 2)

    def get_boundary_points(self, num_points: int = 64) -> List[Tuple[int, int]]:
        """
        Generate points along the ellipse boundary.

        Useful for polygon drawing or path analysis.

        Args:
            num_points: Number of points to generate around ellipse

        Returns:
            List of (x, y) integer tuples forming ellipse boundary
        """
        points = []
        theta_rad = math.radians(self.rotation_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points

            # Parametric ellipse (before rotation)
            x_local = self.semi_major * math.cos(angle)
            y_local = self.semi_minor * math.sin(angle)

            # Apply rotation
            x_rot = x_local * cos_t - y_local * sin_t
            y_rot = x_local * sin_t + y_local * cos_t

            # Translate to center
            x_final = int(self.center_px + x_rot)
            y_final = int(self.center_py + y_rot)

            points.append((x_final, y_final))

        return points

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'name': self.name,
            'center_px': self.center_px,
            'center_py': self.center_py,
            'semi_major': self.semi_major,
            'semi_minor': self.semi_minor,
            'rotation_deg': self.rotation_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurningZone':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            name=data['name'],
            center_px=data['center_px'],
            center_py=data['center_py'],
            semi_major=data['semi_major'],
            semi_minor=data['semi_minor'],
            rotation_deg=data.get('rotation_deg', 0.0),
        )


@dataclass
class TurningZoneConfig:
    """
    Configuration for creating turning zones.

    Stretch factors compensate for camera perspective distortion:
    - stretch_y > 1.0: Vertical stretch for tilted camera (typical: 1.2-1.5)
    - stretch_x: Horizontal stretch (usually 1.0 for aligned cameras)

    Zone radii are specified in pixels directly for simplicity.

    Attributes:
        start_zone_radius: Base radius for START zone (pixels)
        gate2_zone_radius: Base radius for GATE2 zone (pixels)
        stretch_x: Horizontal stretch factor (default: 1.0)
        stretch_y: Vertical compression factor for side-view camera (default: 5.0)
        start_zone_rotation: Rotation of START zone ellipse (degrees)
        gate2_zone_rotation: Rotation of GATE2 zone ellipse (degrees)
    """
    start_zone_radius: float = 150.0
    gate2_zone_radius: float = 180.0
    stretch_x: float = 1.0
    stretch_y: float = 5.0  # Heavy horizontal stretch for side-view camera
    start_zone_rotation: float = 0.0
    gate2_zone_rotation: float = 0.0

    @classmethod
    def default(cls) -> 'TurningZoneConfig':
        """Create default configuration."""
        return cls()

    @classmethod
    def for_overhead_camera(cls) -> 'TurningZoneConfig':
        """Configuration for nearly overhead camera (less distortion)."""
        return cls(stretch_y=1.1)

    @classmethod
    def for_tilted_camera(cls) -> 'TurningZoneConfig':
        """Configuration for tilted camera (more distortion)."""
        return cls(stretch_y=1.5)

    @classmethod
    def small_zones(cls) -> 'TurningZoneConfig':
        """Configuration with smaller, tighter zones."""
        return cls(start_zone_radius=80.0, gate2_zone_radius=100.0)

    @classmethod
    def large_zones(cls) -> 'TurningZoneConfig':
        """Configuration with larger, more generous zones."""
        return cls(start_zone_radius=200.0, gate2_zone_radius=220.0)


@dataclass
class TurningZoneSet:
    """
    Container for both turning zones in a Figure-8 drill.

    Provides convenience methods for checking zone containment.

    Attributes:
        start_zone: Elliptical zone around START cone
        gate2_zone: Elliptical zone around GATE2 midpoint
        config: Configuration used to create these zones
    """
    start_zone: TurningZone
    gate2_zone: TurningZone
    config: TurningZoneConfig = field(default_factory=TurningZoneConfig)

    def get_zone_at_point(self, x: float, y: float) -> Optional[str]:
        """
        Get the name of the zone containing this point.

        Checks START zone first, then GATE2.

        Args:
            x: X-coordinate (pixels)
            y: Y-coordinate (pixels)

        Returns:
            "START", "GATE2", or None if point is not in any zone
        """
        if self.start_zone.contains_point(x, y):
            return "START"
        if self.gate2_zone.contains_point(x, y):
            return "GATE2"
        return None

    def is_in_turning_zone(self, x: float, y: float) -> bool:
        """Check if point is in any turning zone."""
        return self.get_zone_at_point(x, y) is not None

    def get_all_zones(self) -> List[TurningZone]:
        """Get list of all zones."""
        return [self.start_zone, self.gate2_zone]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'start_zone': self.start_zone.to_dict(),
            'gate2_zone': self.gate2_zone.to_dict(),
            'config': {
                'start_zone_radius': self.config.start_zone_radius,
                'gate2_zone_radius': self.config.gate2_zone_radius,
                'stretch_x': self.config.stretch_x,
                'stretch_y': self.config.stretch_y,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurningZoneSet':
        """Create from dictionary (JSON deserialization)."""
        config_data = data.get('config', {})
        config = TurningZoneConfig(
            start_zone_radius=config_data.get('start_zone_radius', 150.0),
            gate2_zone_radius=config_data.get('gate2_zone_radius', 180.0),
            stretch_x=config_data.get('stretch_x', 1.0),
            stretch_y=config_data.get('stretch_y', 1.3),
        )
        return cls(
            start_zone=TurningZone.from_dict(data['start_zone']),
            gate2_zone=TurningZone.from_dict(data['gate2_zone']),
            config=config,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_turning_zones(
    layout: 'Figure8Layout',
    config: Optional[TurningZoneConfig] = None
) -> TurningZoneSet:
    """
    Factory function to create turning zones from a Figure8Layout.

    Creates two elliptical zones:
    - START zone: Centered on start_cone position
    - GATE2 zone: Centered on midpoint of gate2_left and gate2_right

    The ellipse axes are determined by the base radius and stretch factors:
    - semi_major = radius * stretch_x (horizontal)
    - semi_minor = radius * stretch_y (vertical, typically larger for tilted camera)

    Example:
        from f8_loss.detection import load_cone_annotations
        from f8_loss.detection.turning_zones import create_turning_zones

        layout = load_cone_annotations(parquet_dir)
        zones = create_turning_zones(layout)

        if zones.is_in_turning_zone(ball_x, ball_y):
            print(f"Ball in zone: {zones.get_zone_at_point(ball_x, ball_y)}")

    Args:
        layout: Figure8Layout with cone positions from JSON annotations
        config: TurningZoneConfig for radii and stretch factors (uses default if None)

    Returns:
        TurningZoneSet containing START and GATE2 zones
    """
    if config is None:
        config = TurningZoneConfig.default()

    # START zone: centered on start cone
    # Note: semi_major is the LARGER axis, semi_minor is the SMALLER axis
    # For side-view camera: horizontal (X) should be larger, vertical (Y) compressed
    # stretch_y > 1 means we COMPRESS vertical by dividing
    start_zone = TurningZone(
        name="START",
        center_px=layout.start_cone.px,
        center_py=layout.start_cone.py,
        semi_major=config.start_zone_radius * config.stretch_x,  # Horizontal (wider)
        semi_minor=config.start_zone_radius / config.stretch_y,  # Vertical (compressed)
        rotation_deg=config.start_zone_rotation,
    )

    # GATE2 zone: centered on gate2 midpoint
    gate2_center = layout.gate2_center
    gate2_zone = TurningZone(
        name="GATE2",
        center_px=gate2_center[0],
        center_py=gate2_center[1],
        semi_major=config.gate2_zone_radius * config.stretch_x,  # Horizontal (wider)
        semi_minor=config.gate2_zone_radius / config.stretch_y,  # Vertical (compressed)
        rotation_deg=config.gate2_zone_rotation,
    )

    return TurningZoneSet(
        start_zone=start_zone,
        gate2_zone=gate2_zone,
        config=config,
    )


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

# Default colors (BGR format for OpenCV)
START_ZONE_COLOR = (200, 200, 0)      # Teal/Cyan
GATE2_ZONE_COLOR = (200, 100, 200)    # Purple/Magenta
ZONE_HIGHLIGHT_COLOR = (0, 255, 255)  # Bright Yellow


def draw_turning_zone(
    frame: np.ndarray,
    zone: TurningZone,
    color: Tuple[int, int, int] = START_ZONE_COLOR,
    alpha: float = 0.25,
    x_offset: int = 0,
    highlight: bool = False,
    highlight_color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> None:
    """
    Draw a turning zone ellipse on a video frame with transparency.

    Uses cv2.ellipse() for the shape and cv2.addWeighted() for transparency.

    Args:
        frame: Video frame to draw on (modified in place)
        zone: TurningZone to draw
        color: BGR color for normal state
        alpha: Transparency (0.0 = invisible, 1.0 = opaque)
        x_offset: Horizontal offset for sidebar (matches existing pattern)
        highlight: If True, use highlight_color and increased opacity
        highlight_color: Color when highlighted (default: bright yellow)
        thickness: Border thickness when not filled
    """
    if not HAS_CV2:
        return

    # Determine drawing parameters
    draw_color = highlight_color if highlight and highlight_color else color
    if highlight and highlight_color is None:
        draw_color = ZONE_HIGHLIGHT_COLOR
    draw_alpha = min(alpha + 0.15, 0.6) if highlight else alpha

    # Calculate ellipse center with offset
    center = (int(zone.center_px) + x_offset, int(zone.center_py))
    axes = (int(zone.semi_major), int(zone.semi_minor))
    angle = zone.rotation_deg

    # Create overlay for transparency
    overlay = frame.copy()

    # Draw filled ellipse on overlay
    cv2.ellipse(overlay, center, axes, angle, 0, 360, draw_color, -1)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, draw_alpha, frame, 1 - draw_alpha, 0, frame)

    # Draw ellipse border (always visible)
    border_thickness = thickness + 1 if highlight else thickness
    cv2.ellipse(frame, center, axes, angle, 0, 360, draw_color, border_thickness)


def draw_turning_zones(
    frame: np.ndarray,
    zones: TurningZoneSet,
    ball_position: Optional[Tuple[float, float]],
    x_offset: int = 0,
    start_color: Tuple[int, int, int] = START_ZONE_COLOR,
    gate2_color: Tuple[int, int, int] = GATE2_ZONE_COLOR,
    highlight_color: Tuple[int, int, int] = ZONE_HIGHLIGHT_COLOR,
    alpha: float = 0.25,
) -> Optional[str]:
    """
    Draw both turning zones with ball-in-zone highlighting.

    Args:
        frame: Video frame to draw on
        zones: TurningZoneSet containing both zones
        ball_position: (x, y) of ball center, or None if not detected
        x_offset: Sidebar offset
        start_color: Color for START zone
        gate2_color: Color for GATE2 zone
        highlight_color: Color when ball is inside zone
        alpha: Base transparency

    Returns:
        Name of zone containing ball ("START", "GATE2", or None)
    """
    if not HAS_CV2:
        return None

    # Determine if ball is in any zone
    active_zone = None
    if ball_position is not None:
        active_zone = zones.get_zone_at_point(ball_position[0], ball_position[1])

    # Draw START zone
    draw_turning_zone(
        frame,
        zones.start_zone,
        color=start_color,
        alpha=alpha,
        x_offset=x_offset,
        highlight=(active_zone == "START"),
        highlight_color=highlight_color,
    )

    # Draw GATE2 zone
    draw_turning_zone(
        frame,
        zones.gate2_zone,
        color=gate2_color,
        alpha=alpha,
        x_offset=x_offset,
        highlight=(active_zone == "GATE2"),
        highlight_color=highlight_color,
    )

    return active_zone
