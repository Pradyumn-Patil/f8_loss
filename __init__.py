"""
Ball Control Detection System for Figure-8 Cone Drills.

A modular system for detecting when a player loses control of the ball
during Figure-8 cone drill exercises.

Package Structure:
    f8_loss/
    ├── detection/     # Loss of control calculation logic
    ├── annotation/    # Cone annotation and visualization tools
    └── video/         # Video generation with loss events marked

Cone Setup:
    [START] ---- [PAIR1_L] [PAIR1_R] ---- [PAIR2_L] [PAIR2_R]

    - START: Single cone where player starts
    - PAIR1 (Gate 1): First pair of cones with gap
    - PAIR2 (Gate 2): Second pair of cones with gap

Quick Start:
    from f8_loss import detect_ball_control, load_parquet_data, export_to_csv

    # Load data
    ball_df = load_parquet_data("ball.parquet")
    pose_df = load_parquet_data("pose.parquet")
    cone_df = load_parquet_data("cone.parquet")

    # Detect
    result = detect_ball_control(ball_df, pose_df, cone_df)

    # Export
    export_to_csv(result, "output.csv")

Classes:
    - AppConfig: Main configuration container
    - Figure8DrillConfig: Figure-8 specific drill settings
    - BallControlDetector: Core detection class
    - Figure8ConeDetector: Cone role detection and gate tracking
    - CSVExporter: CSV export functionality
    - DrillVisualizer: Debug video visualization

Data Structures:
    - FrameData: Per-frame analysis data
    - LossEvent: A detected loss-of-control event
    - GatePassage: A gate passage event
    - ConeRole: Cone role assignment
    - DetectionResult: Complete detection output
    - ControlState: Ball control state enum
    - DrillPhase: Current phase in drill
    - DrillDirection: Forward/backward direction
"""

# =============================================================================
# Re-export from detection module (backwards compatibility)
# =============================================================================

# Configuration
from .detection.config import (
    AppConfig,
    Figure8DrillConfig,
    DetectionConfig,
    PathConfig,
    VisualizationConfig,
    DetectionMode,
)

# Data structures
from .detection.data_structures import (
    ControlState,
    EventType,
    FrameData,
    LossEvent,
    DetectionResult,
    DrillPhase,
    DrillDirection,
    GatePassage,
    ConeRole,
    # Cone annotation structures
    ConeAnnotation,
    Figure8Layout,
)

# Data loading
from .detection.data_loader import (
    load_parquet_data,
    load_all_data,
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    validate_data_alignment,
    ANKLE_KEYPOINTS,
    # JSON cone loading
    load_cone_annotations,
    EXPECTED_CONE_ROLES,
    # Video metadata
    get_video_fps,
)

# Figure-8 cone detection
from .detection.figure8_cone_detector import Figure8ConeDetector

# Detection
from .detection.ball_control_detector import (
    BallControlDetector,
    detect_ball_control,
)

# Export
from .detection.csv_exporter import (
    CSVExporter,
    export_to_csv,
)

# =============================================================================
# Re-export from annotation module (optional - handles missing OpenCV)
# =============================================================================
try:
    from .annotation.drill_visualizer import DrillVisualizer
    from .annotation.cone_annotator import ConeAnnotator
    _HAS_VISUALIZER = True
except ImportError:
    _HAS_VISUALIZER = False
    DrillVisualizer = None
    ConeAnnotator = None

# =============================================================================
# Re-export from video module (optional - handles missing OpenCV)
# =============================================================================
try:
    from .video.annotate_with_json_cones import (
        annotate_video_with_json_cones,
        convert_to_h264,
        get_available_videos,
    )
    from .video.annotate_videos import annotate_video
    _HAS_VIDEO = True
except ImportError:
    _HAS_VIDEO = False
    annotate_video_with_json_cones = None
    convert_to_h264 = None
    get_available_videos = None
    annotate_video = None


__version__ = "0.3.0"

__all__ = [
    # Version
    '__version__',
    # Config
    'AppConfig',
    'Figure8DrillConfig',
    'DetectionConfig',
    'PathConfig',
    'VisualizationConfig',
    'DetectionMode',
    # Data structures
    'ControlState',
    'EventType',
    'FrameData',
    'LossEvent',
    'DetectionResult',
    'DrillPhase',
    'DrillDirection',
    'GatePassage',
    'ConeRole',
    # Cone annotation structures
    'ConeAnnotation',
    'Figure8Layout',
    # Data loading
    'load_parquet_data',
    'load_all_data',
    'extract_ankle_positions',
    'get_closest_ankle_per_frame',
    'validate_data_alignment',
    'ANKLE_KEYPOINTS',
    # JSON cone loading
    'load_cone_annotations',
    'EXPECTED_CONE_ROLES',
    # Video metadata
    'get_video_fps',
    # Figure-8 detection
    'Figure8ConeDetector',
    # Detection
    'BallControlDetector',
    'detect_ball_control',
    # Export
    'CSVExporter',
    'export_to_csv',
    # Visualization (optional)
    'DrillVisualizer',
    'ConeAnnotator',
    # Video (optional)
    'annotate_video_with_json_cones',
    'convert_to_h264',
    'get_available_videos',
    'annotate_video',
]
