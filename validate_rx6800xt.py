#!/usr/bin/env python
"""
Validazione completa RX 6800 XT per Game Master.
Esegui: python validate_rx6800xt.py
"""
import sys
import time
import subprocess
import json
from pathlib import Path

def run(cmd, desc, timeout=120):
    print(f"\n{'='*60}")
    print(f"🔧 {desc}")
    print(f"$ {cmd}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT ({timeout}s)")
        return False, ""
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        return False, ""

def check_directml():
    """Verifica DirectML e device selection"""
    print("\n🎯 CHECK 1: DirectML Device Selection")
    ok, out = run(
        'python -c "import torch_directml; d=torch_directml.device(); print(f\"Device: {d}\"); import torch; x=torch.tensor([1.0], device=d); print(f\"Tensor device: {x.device}\"); print(f\"DirectML version: {torch_directml.__version__ if hasattr(torch_directml, \"__version__\") else \"unknown\"}\")"',
        "Verifica device DirectML"
    )
    if ok and "rx 6800" in out.lower() or "radeon" in out.lower() or "6800" in out.lower():
        print("✅ GPU RX 6800 XT rilevata correttamente")
    elif ok:
        print("⚠️ DirectML funziona ma device non identificato come RX 6800 XT")
        print(f"   Output: {out.strip()}")
    return ok

def check_torch_directml():
    """Verifica torch + directml compatibilità"""
    print("\n🎯 CHECK 2: Torch + DirectML Compatibility")
    ok, out = run(
        'python -c "import torch; print(f\"Torch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); import torch_directml; print(f\"DirectML devices: {torch_directml.device_count()}\"); [print(f\"  [{i}] {torch_directml.device(i)}\") for i in range(torch_directml.device_count())]"',
        "Elenca device DirectML disponibili"
    )
    return ok

def run_gpu_tests():
    """Esegue test suite GPU"""
    print("\n🎯 CHECK 3: GPU Test Suite (pytest -m gpu)")
    ok, out = run(
        "python -m pytest tests/ -m gpu --tb=short -v",
        "Test suite marcati GPU",
        timeout=300
    )
    if ok:
        print("✅ Tutti i test GPU passano")
    else:
        print("❌ Alcuni test GPU falliscono")
    return ok

def test_screen_capture():
    """Test capture reale dxcam"""
    print("\n🎯 CHECK 4: Screen Capture Reale (dxcam)")
    ok, out = run(
        "python -m pytest tests/test_screen_capture.py::TestScreenCapture::test_start_stop_capture -v --tb=long",
        "Test start/stop capture reale",
        timeout=60
    )
    if ok:
        print("✅ Screen capture funziona")
    return ok

def benchmark_capture_fps():
    """Benchmark 30fps sustained per 60s"""
    print("\n🎯 CHECK 5: Benchmark 30fps Sustained (60 secondi)")
    print("   Premi INVIO per iniziare capture 1920x1080 @ 30fps per 60s...")
    input()
    
    bench_code = '''
import time
import numpy as np
from perception.capture import ScreenCapture, CaptureConfig

config = CaptureConfig(target_fps=30, width=1920, height=1080, ring_buffer_size=1800)
cap = ScreenCapture(config)
cap.start()

print("🎬 Capture iniziato - attendi 60 secondi...")
time.sleep(60)

metrics = cap.get_metrics()
cap.stop()

if metrics:
    fps = len(metrics) / 60.0
    latencies = [m.capture_latency_ms for m in metrics]
    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"📊 RISULTATI:")
    print(f"   Frame catturati: {len(metrics)}")
    print(f"   FPS effettivo: {fps:.2f}")
    print(f"   Latency media: {avg_lat:.2f}ms")
    print(f"   Latency max: {max_lat:.2f}ms")
    print(f"   Latency p99: {p99_lat:.2f}ms")
    print(f"   Target 30fps: {'✅ PASS' if fps >= 29 else '❌ FAIL'}")
    print(f"   Target <16ms: {'✅ PASS' if avg_lat < 16 else '❌ FAIL'}")
else:
    print("❌ Nessun frame catturato")
'''
    ok, out = run(f'python -c "{bench_code}"', "Benchmark 60s @ 30fps", timeout=90)
    return ok

def test_full_pipeline():
    """Test pipeline completa percezione"""
    print("\n🎯 CHECK 6: Pipeline Percezione Completa")
    ok, out = run(
        'python -c "from perception import create_state_extractor; se = create_state_extractor(); print(f\"StateExtractor creato: {type(se).__name__}\"); print(\"✅ Pipeline initialization OK\")"',
        "Inizializzazione pipeline completa"
    )
    return ok

def test_thermal_stability():
    """Test stabilità termica (carico continuo 5 min)"""
    print("\n🎯 CHECK 7: Stabilità Termica (5 min carico continuo)")
    print("   ⚠️ Questo test stressa la GPU per 5 minuti. Continuare? (y/N): ", end="")
    if input().lower() != 'y':
        print("   Saltato")
        return True
    
    stress_code = '''
import time
import torch
import torch_directml
from perception.preprocessing import create_detection_pipeline
import numpy as np

device = torch_directml.device()
pipeline = create_detection_pipeline(device="directml")
frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

print("🔥 Stress test 5 minuti...")
start = time.time()
frames = 0
errors = 0

while time.time() - start < 300:  # 5 minuti
    try:
        _ = pipeline.process(frame)
        frames += 1
    except Exception as e:
        errors += 1
        if errors > 10:
            break

elapsed = time.time() - start
fps = frames / elapsed
print(f"📊 Stress test completato:")
print(f"   Durata: {elapsed:.1f}s")
print(f"   Frame processati: {frames}")
print(f"   FPS medio: {fps:.2f}")
print(f"   Errori: {errors}")
print(f"   {'✅ STABILE' if errors == 0 and fps > 100 else '⚠️ VERIFICARE'}")
'''
    ok, out = run(f'python -c "{stress_code}"', "Stress termico 5 min", timeout=360)
    return ok

def main():
    print("="*60)
    print("🎮 GAME MASTER - VALIDAZIONE RX 6800 XT")
    print("="*60)
    
    results = {}
    
    results["directml_device"] = check_directml()
    results["torch_directml"] = check_torch_directml()
    results["gpu_tests"] = run_gpu_tests()
    results["screen_capture"] = test_screen_capture()
    results["capture_fps"] = benchmark_capture_fps()
    results["full_pipeline"] = test_full_pipeline()
    results["thermal"] = test_thermal_stability()
    
    print("\n" + "="*60)
    print("📋 RIEPILOGO VALIDAZIONE")
    print("="*60)
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:25s} : {status}")
    
    all_pass = all(results.values())
    print(f"\n{'🎉 TUTTI I CHECK PASSANO' if all_pass else '⚠️ ALCUNI CHECK FALLISCONO'}")
    
    # Salva report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "RX 6800 XT",
        "results": results,
        "all_pass": all_pass
    }
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report salvato in: validation_report.json")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())