# Fix: Ball Behind Detection Not Triggering When Player Stops

## Problem
At 34.5s in arjun_mital video, player stops moving while ball is 100-142px behind them. Detection fails because `direction=STATIC` causes ball-behind check to be skipped.

## Root Cause
In `detection/ball_control_detector.py` at line 638-639:
```python
if hip_pixel_pos is None or player_direction is None:
    return False, None  # SKIPS check when direction is None/STATIC
```

When player STOPS (hip movement < 3px/frame), direction becomes None, and the entire ball-behind check is skipped even though ball is clearly behind the player.

## Data Evidence
```
Frame 1026-1031: Direction=RIGHT, Behind=YES (6 frames)
Frame 1032-1044: Direction=STATIC, Behind=SKIPPED (13 frames - ball is 100-142px behind but ignored!)
Frame 1045+:     Direction=LEFT, Behind=NO (player recovering toward ball)
```

The detection needs 14 consecutive frames of behind=True, but only gets 6 before direction goes STATIC.

## Fix Required

### Step 1: Add direction history tracking
In `BallControlDetector.__init__()` around line 99, add:
```python
self._direction_history: deque = deque(maxlen=15)  # Track recent directions
```

### Step 2: Store directions in history
In `_analyze_frame()` around line 328, after calculating `player_direction`, add:
```python
self._direction_history.append(player_direction)
```

### Step 3: Use previous direction when current is None
In `_analyze_frame()` around line 328, modify to fall back to last known direction:
```python
# If direction is unclear, use the most recent non-None direction
if player_direction is None:
    for d in reversed(self._direction_history):
        if d is not None:
            player_direction = d
            break
```

### Step 4: Update detect_loss() to not skip when using fallback direction
The line 638-639 check should still allow detection if we have a fallback direction from history.

## Expected Result
- Frames 1026-1044: All 19 frames register as Behind=True
- Loss event created at ~34.5s (frame 1039, after 14 consecutive behind frames)
- Event lasts until recovery at ~35.8s
- Event duration: ~1.3s (40 frames) - passes the 15-frame minimum filter

## Verification
```bash
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM" && PYTHONPATH="." python f8_loss/run_detection.py arjun_mital
```

Expected: 1 loss event at ~34.5s with type `ball_behind`

## Files to Modify
- `detection/ball_control_detector.py`
  - Line ~99: Add `_direction_history` deque
  - Line ~328: Store direction in history and implement fallback logic
  - Line ~638-639: Allow detection to proceed with fallback direction
