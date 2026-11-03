#!/usr/bin/env python
"""Quick GPU check - esegui in 30 secondi"""
import subprocess, sys

checks = [
    ("DirectML device", 'python -c "import torch_directml; print(torch_directml.device())"'),
    ("Torch + DirectML", 'python -c "import torch, torch_directml; print(f\"Torch {torch.__version__}, DML devices: {torch_directml.device_count()}\"); [print(f\"  [{i}] {torch_directml.device(i)}\") for i in range(torch_directml.device_count())]"'),
    ("Preprocessing GPU", 'python -c "from perception.preprocessing import create_detection_pipeline; import numpy as np; p=create_detection_pipeline(device=\"directml\"); import time; f=np.random.randint(0,255,(640,640,3),dtype=np.uint8); t=time.time(); [p.process(f) for _ in range(100)]; print(f\"100 frames: {time.time()-t:.2f}s = {100/(time.time()-t):.1f} FPS\")"'),
    ("Screen capture", 'python -c "from perception.capture import ScreenCapture, CaptureConfig; c=ScreenCapture(CaptureConfig(width=640,height=480)); c.start(); import time; time.sleep(2); f=c.get_frame(); c.stop(); print(f\"Frame shape: {f.shape if f is not None else None}\")"'),
]

print("⚡ QUICK GPU CHECK RX 6800 XT\n")
for name, cmd in checks:
    print(f"🔍 {name}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        print(f"   ✅ {r.stdout.strip()}")
    else:
        print(f"   ❌ {r.stderr.strip()[:200]}")
    print()

print("Fatto! Se tutti ✅, hardware pronto per Game Master.")