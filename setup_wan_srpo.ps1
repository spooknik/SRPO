# Complete setup script for WAN SRPO on Windows (RTX 6000 Pro 96GB)
# Usage: .\setup_wan_srpo.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "WAN SRPO Complete Setup (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion"

Write-Host ""
Write-Host "Step 1/6: Installing PyTorch with CUDA 12.4..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "Step 2/6: Installing Flash Attention 2..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "Note: Flash Attention may require Visual Studio Build Tools" -ForegroundColor Yellow
pip install flash-attn==2.7.0 --no-build-isolation

Write-Host ""
Write-Host "Step 3/6: Installing diffusers from main (WAN support)..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
pip install git+https://github.com/huggingface/diffusers.git

Write-Host ""
Write-Host "Step 4/6: Installing base dependencies from pyproject.toml..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
pip install -e .

Write-Host ""
Write-Host "Step 5/6: Installing HPSv2 reward model..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
if (-Not (Test-Path "hpsv2")) {
    git clone https://github.com/tgxs002/HPSv2.git hpsv2
    Set-Location hpsv2
    pip install -e .
    Set-Location ..
    Write-Host "✓ HPSv2 installed" -ForegroundColor Green
} else {
    Write-Host "✓ HPSv2 already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 6/6: Installing additional WAN requirements..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
pip install PyYAML

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
try {
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
} catch {
    Write-Host "Flash Attention: Installation may have failed (optional)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Download WAN model:" -ForegroundColor White
Write-Host "   huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir ./data/wan2.2-t2v-a14b" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Download HPS reward model:" -ForegroundColor White
Write-Host "   huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt" -ForegroundColor Gray
Write-Host "   huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./data/hps_ckpt" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Convert your prompts:" -ForegroundColor White
Write-Host "   python scripts/utils/convert_yaml_to_json.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Run training:" -ForegroundColor White
Write-Host "   You'll need to use WSL or adapt the bash script to PowerShell" -ForegroundColor Gray
