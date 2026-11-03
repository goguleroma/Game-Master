# Quick GPU Check per PowerShell - salva come .ps1 ed esegui
Write-Host "`n== QUICK GPU CHECK RX 6800 XT =="

$checks = @(
    @{Name="DirectML device"; Cmd='python -c "import torch_directml; print(torch_directml.device())"'}
    @{Name="Torch + DirectML"; Cmd='python -c "import torch, torch_directml; print(f\"Torch {torch.__version__}, DML devices: {torch_directml.device_count()}\"); [print(f\"  [{i}] {torch_directml.device(i)}\") for i in range(torch_directml.device_count())]"'}
    @{Name="Preprocessing GPU"; Cmd='python -c "from perception.preprocessing import create_detection_pipeline; import numpy as np; p=create_detection_pipeline(device=`"directml`"); import time; f=np.random.randint(0,255,(640,640,3),dtype=np.uint8); t=time.time(); [p.process(f) for _ in range(100)]; print(f\"100 frames: {time.time()-t:.2f}s = {100/(time.time()-t):.1f} FPS\")"'}
    @{Name="Screen capture"; Cmd='python -c "from perception.capture import ScreenCapture, CaptureConfig; c=ScreenCapture(CaptureConfig(width=640,height=480)); c.start(); import time; time.sleep(2); f=c.get_frame(); c.stop(); print(f\"Frame shape: {f.shape if f is not None else None}\")"'}
)

foreach ($check in $checks) {
    Write-Host "`nChecking: $($check.Name)..."
    try {
        $result = Invoke-Expression $check.Cmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK: $result"
        } else {
            Write-Host "  FAIL (exit code: $LASTEXITCODE)"
        }
    } catch {
        Write-Host "  ERROR: $($_.Exception.Message)"
    }
}

Write-Host "`nDone! Se tutti OK, hardware pronto per Game Master."