#!/usr/bin/env python
"""Check DirectML device"""
import directml
device = directml.device()
print(f"DirectML Device: {device}")
print(f"Device count: {directml.device_count()}")
for i in range(directml.device_count()):
    print(f"  [{i}] {directml.device(i)}")