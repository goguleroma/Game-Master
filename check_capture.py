#!/usr/bin/env python
"""Check screen capture"""
import time
from perception.capture import ScreenCapture, CaptureConfig

c = ScreenCapture(CaptureConfig(width=640, height=480))
c.start()
time.sleep(2)
frame = c.get_frame()
c.stop()

if frame is not None:
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
else:
    print("Frame: None (capture failed)")