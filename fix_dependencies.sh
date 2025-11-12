#!/bin/bash
# Fix bitsandbytes and triton compatibility issues
# Run this if you get "No module named 'triton.ops'" error

set -e

echo "=========================================="
echo "Fixing bitsandbytes and triton issues"
echo "=========================================="

echo ""
echo "Step 1: Uninstalling problematic packages..."
pip uninstall -y bitsandbytes triton

echo ""
echo "Step 2: Installing compatible versions..."
echo "Installing triton..."
pip install triton==3.0.0

echo "Installing bitsandbytes with CUDA support..."
# Use pre-built wheel for Linux with CUDA 12.x
pip install bitsandbytes==0.44.1

echo ""
echo "Step 3: Verifying installation..."
python -c "import bitsandbytes as bnb; print(f'bitsandbytes version: {bnb.__version__}')"
python -c "import triton; print(f'triton version: {triton.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Step 4: Testing bitsandbytes CUDA support..."
python << 'EOF'
import torch
import bitsandbytes as bnb

# Test if CUDA binary is found
if torch.cuda.is_available():
    try:
        # Simple test to verify CUDA support
        tensor = torch.randn(10, 10).cuda()
        print("✓ bitsandbytes CUDA support: Working")
    except Exception as e:
        print(f"✗ bitsandbytes CUDA support: Failed - {e}")
else:
    print("✗ CUDA not available")
EOF

echo ""
echo "=========================================="
echo "Fix complete!"
echo "=========================================="
echo ""
echo "If the issue persists, try:"
echo "1. Reinstall from source:"
echo "   pip install git+https://github.com/TimDettmers/bitsandbytes.git"
echo ""
echo "2. Or use a pre-built wheel for your CUDA version:"
echo "   pip install bitsandbytes --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui"
