#!/bin/bash
# Complete setup script for WAN SRPO on RTX 6000 Pro (96GB)
# Usage: bash setup_wan_srpo.sh

set -e  # Exit on error

echo "=========================================="
echo "WAN SRPO Complete Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VERSION}"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher required"
    exit 1
fi

echo ""
echo "Step 1/6: Installing PyTorch with CUDA 12.4..."
echo "=========================================="
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Step 2/6: Installing Flash Attention 2..."
echo "=========================================="
pip install flash-attn==2.7.0 --no-build-isolation

echo ""
echo "Step 3/6: Installing diffusers from main (WAN support)..."
echo "=========================================="
pip install git+https://github.com/huggingface/diffusers.git

echo ""
echo "Step 4/6: Installing base dependencies from pyproject.toml..."
echo "=========================================="
pip install -e .

echo ""
echo "Step 5/6: Installing HPSv2 reward model..."
echo "=========================================="
if [ ! -d "hpsv2" ]; then
    git clone https://github.com/tgxs002/HPSv2.git hpsv2
    cd hpsv2
    pip install -e .
    cd ..
    echo "✓ HPSv2 installed"
else
    echo "✓ HPSv2 already exists"
fi

echo ""
echo "Step 6/6: Installing additional WAN requirements..."
echo "=========================================="
pip install PyYAML  # For prompts.yaml conversion

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA version: Not available"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "Flash Attention: Installation may have failed"

echo ""
echo "Next steps:"
echo "1. Download WAN model:"
echo "   huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir ./data/wan2.2-t2v-a14b"
echo ""
echo "2. Download HPS reward model:"
echo "   huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt"
echo "   huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ./data/hps_ckpt"
echo ""
echo "3. Convert your prompts:"
echo "   bash scripts/utils/convert_prompts.sh"
echo ""
echo "4. Run training:"
echo "   bash scripts/finetune/SRPO_training_wan_t2i_single_gpu.sh"
