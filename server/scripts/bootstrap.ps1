<#
    Bootstrap script for GhostTab dev environment
    Usage:
      powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
#>

# Stop on first error
$ErrorActionPreference = "Stop"

Write-Host "🚀 Bootstrapping GhostTab development environment..."

# -------------------------------
# 1) Create virtual environment
# -------------------------------
if (-Not (Test-Path "venv")) {
    Write-Host "📦 Creating Python virtual environment..."
    py -m venv venv
} else {
    Write-Host "✅ venv already exists, skipping creation."
}

# Activate the venv
Write-Host "🔗 Activating venv..."
& .\venv\Scripts\Activate.ps1

# -------------------------------
# 2) Upgrade pip + install deps
# -------------------------------
Write-Host "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    Write-Host "📚 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
} else {
    Write-Host "⚠️ No requirements.txt found, skipping dependency install."
}

# -------------------------------
# 3) Export ONNX sentiment model
# -------------------------------
$exportScript = "scripts/export_sentiment.py"
if (Test-Path $exportScript) {
    Write-Host "🧠 Exporting ONNX sentiment model..."
    python $exportScript
} else {
    Write-Host "⚠️ $exportScript not found, skipping model export."
}

Write-Host "✅ Bootstrap complete! Run your server with:"
Write-Host "   uvicorn app:app --reload --port 8000"
