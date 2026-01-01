# Detection module - Loss of control calculation logic
"""
Core detection logic for Figure-8 ball control analysis.

This module contains:
- BallControlDetector: Main detection engine with detect_loss() method
- Figure8ConeDetector: Cone role identification and gate tracking
- Data structures: ControlState, LossEvent, FrameData, etc.
- Configuration: AppConfig, DetectionConfig
- Data loading: load_cone_annotations, load_parquet_data
- Export: CSVExporter
"""

from .ball_control_detector import BallControlDetector, detect_ball_control
from .figure8_cone_detector import Figure8ConeDetector
from .data_structures import (
    ControlState, EventType, DrillPhase, DrillDirection,
    ConeAnnotation, Figure8Layout, FrameData, LossEvent,
    GatePassage, ConeRole, DetectionResult
)
from .config import (
    AppConfig, DetectionConfig, Figure8DrillConfig,
    PathConfig, VisualizationConfig, DetectionMode
)
from .data_loader import (
    load_cone_annotations, load_parquet_data, load_all_data,
    extract_ankle_positions, get_closest_ankle_per_frame,
    validate_data_alignment, get_frame_data, get_video_fps
)
from .csv_exporter import CSVExporter, export_to_csv
from .turning_zones import (
    TurningZone, TurningZoneConfig, TurningZoneSet,
    create_turning_zones, draw_turning_zone, draw_turning_zones,
    START_ZONE_COLOR, GATE2_ZONE_COLOR, ZONE_HIGHLIGHT_COLOR,
)

__all__ = [
    # Detector classes
    'BallControlDetector', 'detect_ball_control',
    'Figure8ConeDetector',
    # Data structures
    'ControlState', 'EventType', 'DrillPhase', 'DrillDirection',
    'ConeAnnotation', 'Figure8Layout', 'FrameData', 'LossEvent',
    'GatePassage', 'ConeRole', 'DetectionResult',
    # Configuration
    'AppConfig', 'DetectionConfig', 'Figure8DrillConfig',
    'PathConfig', 'VisualizationConfig', 'DetectionMode',
    # Data loading
    'load_cone_annotations', 'load_parquet_data', 'load_all_data',
    'extract_ankle_positions', 'get_closest_ankle_per_frame',
    'validate_data_alignment', 'get_frame_data', 'get_video_fps',
    # Export
    'CSVExporter', 'export_to_csv',
    # Turning zones
    'TurningZone', 'TurningZoneConfig', 'TurningZoneSet',
    'create_turning_zones', 'draw_turning_zone', 'draw_turning_zones',
    'START_ZONE_COLOR', 'GATE2_ZONE_COLOR', 'ZONE_HIGHLIGHT_COLOR',
]
