#!/usr/bin/env python
"""Check preprocessing GPU performance"""
import time
import numpy as np
from perception.preprocessing import create_detection_pipeline

p = create_detection_pipeline(device="directml")
f = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(10):
    p.process(f)

# Benchmark
t = time.time()
for _ in range(100):
    p.process(f)
elapsed = time.time() - t
fps = 100 / elapsed
print(f"100 frames: {elapsed:.2f}s = {fps:.1f} FPS")