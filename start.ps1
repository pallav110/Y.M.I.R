# Y.M.I.R AI System PowerShell Launcher
# Run with: powershell -ExecutionPolicy Bypass -File start.ps1

Write-Host "🧠 Y.M.I.R AI Emotion Detection System" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Blue

# Check if virtual environment exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️  Virtual environment not found, using system Python" -ForegroundColor Yellow
}

# Run the launcher
Write-Host "🚀 Starting Y.M.I.R services..." -ForegroundColor Green
python start_ymir.py

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")