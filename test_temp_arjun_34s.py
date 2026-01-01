"""
Temporary debug script to analyze arjun_mital at 34 seconds.
Investigating why ball_behind loss detection is NOT triggering.

Ground truth: Loss at 34.5-36s, "Overstopped - ball moved up inside G1 while returning from turn"
"""
import sys
sys.path.insert(0, '/Users/pradyumn/Desktop/FOOTBALL data /AIM')

from pathlib import Path
import pandas as pd
from f8_loss.detection import BallControlDetector, AppConfig
from f8_loss.detection.data_loader import load_cone_annotations
from f8_loss.detection.turning_zones import create_turning_zones

# Setup paths
parquet_dir = Path('/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/video_detection_pose_ball_cones/arjun_mital_f8')

# Load data
print("=" * 80)
print("Loading arjun_mital_f8 data...")
print("=" * 80)

ball_df = pd.read_parquet(parquet_dir / 'arjun_mital_f8_football.parquet')
pose_df = pd.read_parquet(parquet_dir / 'arjun_mital_f8_pose.parquet')

# Get frame rate info
fps = 30  # Assume 30fps
target_time = 34.0  # User said 34 sec
target_frame_start = int(target_time * fps)  # Frame 1020
target_frame_end = int(36.0 * fps)  # Frame 1080

print(f"\nTarget time range: {target_time}s - 36.0s")
print(f"Target frame range: {target_frame_start} - {target_frame_end}")

# Load cone annotations and turning zones
layout = load_cone_annotations(parquet_dir / 'cone_annotations.json')
zones = create_turning_zones(layout)

print(f"\nCone layout:")
print(f"  Start cone: ({layout.start_cone.px:.0f}, {layout.start_cone.py:.0f})")
print(f"  Gate1: L=({layout.gate1_left.px:.0f}, {layout.gate1_left.py:.0f}), R=({layout.gate1_right.px:.0f}, {layout.gate1_right.py:.0f})")
print(f"  Gate2: L=({layout.gate2_left.px:.0f}, {layout.gate2_left.py:.0f}), R=({layout.gate2_right.px:.0f}, {layout.gate2_right.py:.0f})")

print(f"\nTurning zones:")
print(f"  START zone: center=({zones.start_zone.center_px:.0f}, {zones.start_zone.center_py:.0f}), semi_major={zones.start_zone.semi_major:.0f}, semi_minor={zones.start_zone.semi_minor:.0f}")
print(f"  GATE2 zone: center=({zones.gate2_zone.center_px:.0f}, {zones.gate2_zone.center_py:.0f}), semi_major={zones.gate2_zone.semi_major:.0f}, semi_minor={zones.gate2_zone.semi_minor:.0f}")

print("\n" + "=" * 80)
print("FRAME-BY-FRAME ANALYSIS (32s - 37s)")
print("=" * 80)

# Analyze frames from 32s to 37s
for frame_id in range(int(32 * fps), int(37 * fps)):
    timestamp = frame_id / fps

    # Get ball position for this frame
    ball_frame = ball_df[ball_df['frame_id'] == frame_id]
    if ball_frame.empty:
        continue

    ball_row = ball_frame.iloc[0]
    ball_x = ball_row.get('pixel_x', ball_row.get('x', None))
    ball_y = ball_row.get('pixel_y', ball_row.get('y', None))

    if ball_x is None or pd.isna(ball_x):
        continue

    # Get pose for this frame
    pose_frame = pose_df[pose_df['frame_id'] == frame_id]

    # Find hip position
    hip_x, hip_y = None, None
    if not pose_frame.empty:
        # Try to find hip keypoint
        hip_row = pose_frame[pose_frame['keypoint_name'].isin(['hip', 'pelvis', 'mid_hip'])]
        if hip_row.empty:
            # Try to compute from left_hip and right_hip
            left_hip = pose_frame[pose_frame['keypoint_name'] == 'left_hip']
            right_hip = pose_frame[pose_frame['keypoint_name'] == 'right_hip']
            if not left_hip.empty and not right_hip.empty:
                lh_x = left_hip.iloc[0].get('pixel_x', left_hip.iloc[0].get('x'))
                rh_x = right_hip.iloc[0].get('pixel_x', right_hip.iloc[0].get('x'))
                lh_y = left_hip.iloc[0].get('pixel_y', left_hip.iloc[0].get('y'))
                rh_y = right_hip.iloc[0].get('pixel_y', right_hip.iloc[0].get('y'))
                if not pd.isna(lh_x) and not pd.isna(rh_x):
                    hip_x = (lh_x + rh_x) / 2
                    hip_y = (lh_y + rh_y) / 2
        else:
            hip_x = hip_row.iloc[0].get('pixel_x', hip_row.iloc[0].get('x'))
            hip_y = hip_row.iloc[0].get('pixel_y', hip_row.iloc[0].get('y'))

    # Check turning zone
    in_zone = zones.get_zone_at_point(ball_x, ball_y)

    # Calculate ball-hip delta
    delta_x = None
    ball_position_relative = None
    if hip_x is not None and not pd.isna(hip_x):
        delta_x = ball_x - hip_x
        if delta_x > 30:
            ball_position_relative = "RIGHT of hip (+)"
        elif delta_x < -30:
            ball_position_relative = "LEFT of hip (-)"
        else:
            ball_position_relative = "NEAR hip"

    # Print every 5 frames or critical frames
    if frame_id % 5 == 0 or (timestamp >= 34.0 and timestamp <= 36.0):
        zone_str = f"IN {in_zone}" if in_zone else "not in zone"
        hip_str = f"({hip_x:.0f}, {hip_y:.0f})" if hip_x is not None else "N/A"
        delta_str = f"{delta_x:+.0f}px" if delta_x is not None else "N/A"
        rel_str = ball_position_relative or "N/A"

        marker = " <<<< CRITICAL" if 34.0 <= timestamp <= 36.0 else ""
        print(f"Frame {frame_id:4d} ({timestamp:5.2f}s): Ball=({ball_x:.0f}, {ball_y:.0f}) Hip={hip_str} Delta_X={delta_str} [{rel_str}] Zone={zone_str}{marker}")

print("\n" + "=" * 80)
print("RUNNING FULL DETECTION...")
print("=" * 80)

# Run full detection
config = AppConfig.for_figure8()
detector = BallControlDetector(config)

result = detector.detect(
    ball_df=ball_df,
    pose_df=pose_df,
    layout=layout,
    video_width=1920,
    video_height=1080,
    fps=fps
)

print(f"\nDetected {len(result.loss_events)} loss events:")
for event in result.loss_events:
    print(f"  - Event #{event.event_id}: {event.event_type.value} at {event.start_timestamp:.2f}s - {event.end_timestamp:.2f}s")
    print(f"    Ball position: ({event.ball_position[0]:.0f}, {event.ball_position[1]:.0f})")

# Check frames around 34s for ball_behind_player flag
print("\n" + "=" * 80)
print("CHECKING FRAME DATA FLAGS (34-36s)")
print("=" * 80)

for frame_data in result.frame_data:
    if 34.0 <= frame_data.timestamp <= 36.0:
        print(f"Frame {frame_data.frame_id} ({frame_data.timestamp:.2f}s):")
        print(f"  ball_behind_player: {frame_data.ball_behind_player}")
        print(f"  player_movement_direction: {frame_data.player_movement_direction}")
        print(f"  in_turning_zone: {frame_data.in_turning_zone}")
        print(f"  control_state: {frame_data.control_state}")
        print(f"  hip: ({frame_data.hip_x}, {frame_data.hip_y})")
        print(f"  ball: ({frame_data.ball_x:.0f}, {frame_data.ball_y:.0f})" if frame_data.ball_x else "  ball: N/A")
        print()
