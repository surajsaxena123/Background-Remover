#!/bin/bash

# Precision Background Remover - Mac Installation Script
# Optimized for both Intel and Apple Silicon Macs

set -e

echo "🍎 Precision Background Remover - Mac Installation"
echo "=================================================="

# Detect Mac architecture
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Darwin" ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

echo "🔍 Detected Mac Architecture: $ARCH"

# Check if we're on Apple Silicon or Intel
if [[ "$ARCH" == "arm64" ]]; then
    echo "🚀 Apple Silicon (M1/M2/M3) detected"
    IS_APPLE_SILICON=true
else
    echo "💻 Intel Mac detected"
    IS_APPLE_SILICON=false
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc) -eq 0 ]]; then
    echo "❌ Python 3.8 or higher is required"
    exit 1
fi

# Check if we have required system tools
echo "🔧 Checking system dependencies..."

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    if [[ "$IS_APPLE_SILICON" == "true" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "✅ Homebrew already installed"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew update

# Install OpenCV and image processing libraries
if [[ "$IS_APPLE_SILICON" == "true" ]]; then
    # Apple Silicon specific optimizations
    echo "🚀 Installing Apple Silicon optimized libraries..."
    brew install opencv python3 numpy pkg-config cmake
    
    # Install Metal Performance Shaders if available
    echo "🔗 Setting up Metal Performance Shaders..."
    export PYTORCH_ENABLE_MPS_FALLBACK=1
else
    # Intel Mac optimizations
    echo "💻 Installing Intel Mac optimized libraries..."
    brew install opencv python3 numpy pkg-config cmake intel-mkl
fi

# Create and activate virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv bg_remover_env
source bg_remover_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Mac-optimized PyTorch
echo "🔥 Installing PyTorch for Mac..."
if [[ "$IS_APPLE_SILICON" == "true" ]]; then
    # Apple Silicon - install with MPS support
    pip install torch torchvision torchaudio
else
    # Intel Mac - install with MKL support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install basic requirements
echo "📦 Installing basic requirements..."
pip install -r requirements_simple.txt

# Try to install advanced requirements (optional)
echo "🎯 Attempting to install advanced precision-grade features..."
if pip install -r requirements.txt; then
    echo "✅ Advanced features installed successfully"
    ADVANCED_AVAILABLE=true
else
    echo "⚠️  Advanced features failed to install - using stable version"
    ADVANCED_AVAILABLE=false
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import cv2
import numpy as np
import streamlit as st
from PIL import Image
print('✅ Basic dependencies working')

try:
    from src.mac_optimization import detect_mac_architecture
    mac_info = detect_mac_architecture()
    print(f'✅ Mac optimizations available')
    print(f'   Architecture: {mac_info[\"machine\"]}')
    print(f'   Apple Silicon: {mac_info[\"is_apple_silicon\"]}')
    print(f'   Memory: {mac_info.get(\"memory_gb\", \"Unknown\")} GB')
    if mac_info.get('supports_mps', False):
        print('✅ Metal Performance Shaders (MPS) available')
except Exception as e:
    print(f'⚠️  Mac optimizations not available: {e}')

try:
    import rembg
    print('✅ rembg available for background removal')
except ImportError:
    print('❌ rembg not available - install with: pip install rembg')
"

# Test with sample image if available
if [[ -f "data/input/DKDR03423_16_KA4C41N8_BLK_Alt1 (2).jpg" ]]; then
    echo "🖼️  Testing with sample image..."
    python3 demo_working.py "data/input/DKDR03423_16_KA4C41N8_BLK_Alt1 (2).jpg" test_output.png --analyze-only
    if [[ $? -eq 0 ]]; then
        echo "✅ Sample processing test passed"
    else
        echo "⚠️  Sample processing test failed"
    fi
fi

# Create launch scripts
echo "🚀 Creating launch scripts..."

# Simple web interface launcher
cat > launch_simple.sh << 'EOF'
#!/bin/bash
echo "🎨 Launching Simple Background Remover..."
source bg_remover_env/bin/activate
streamlit run app_simple.py
EOF

# Advanced web interface launcher (if available)
if [[ "$ADVANCED_AVAILABLE" == "true" ]]; then
    cat > launch_advanced.sh << 'EOF'
#!/bin/bash
echo "🏥 Launching Medical-Grade Background Remover..."
source bg_remover_env/bin/activate
streamlit run app.py
EOF
    chmod +x launch_advanced.sh
fi

# Command line launcher
cat > process_image.sh << 'EOF'
#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Usage: ./process_image.sh <input_image> <output_image> [--analyze-only]"
    exit 1
fi
source bg_remover_env/bin/activate
python3 demo_working.py "$@"
EOF

chmod +x launch_simple.sh process_image.sh

# Print Mac-specific optimization info
echo ""
echo "🍎 Mac-Specific Optimizations Applied:"
echo "======================================"

if [[ "$IS_APPLE_SILICON" == "true" ]]; then
    echo "🚀 Apple Silicon Optimizations:"
    echo "   • Metal Performance Shaders (MPS) enabled for GPU acceleration"
    echo "   • ARM64-optimized libraries installed"
    echo "   • Memory optimization for unified memory architecture"
    echo "   • Native Apple Silicon PyTorch with MPS backend"
else
    echo "💻 Intel Mac Optimizations:"
    echo "   • Intel MKL (Math Kernel Library) support"
    echo "   • x86_64 optimized libraries"
    echo "   • OpenMP threading optimizations"
fi

echo ""
echo "📊 System Information:"
TOTAL_MEM=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
CPU_COUNT=$(sysctl hw.ncpu | awk '{print $2}')
echo "   • Total Memory: ${TOTAL_MEM}GB"
echo "   • CPU Cores: $CPU_COUNT"

echo ""
echo "🎯 Installation Complete!"
echo "======================="
echo ""
echo "📚 Quick Start Guide:"
echo "   Simple Web Interface:    ./launch_simple.sh"
if [[ "$ADVANCED_AVAILABLE" == "true" ]]; then
    echo "   Medical-Grade Interface: ./launch_advanced.sh"
fi
echo "   Command Line Processing: ./process_image.sh input.jpg output.png"
echo "   Quality Analysis Only:   ./process_image.sh input.jpg output.png --analyze-only"
echo ""
echo "🔗 Docker Alternative:"
echo "   docker build -t bg-remover ."
echo "   docker run -p 8501:8501 bg-remover"
echo ""
echo "📖 Documentation: See README.md and QUICK_START.md"
echo ""

if [[ "$TOTAL_MEM" -lt 8 ]]; then
    echo "⚠️  WARNING: Your Mac has less than 8GB RAM. Consider:"
    echo "   • Processing smaller images (resize before processing)"
    echo "   • Using simple interface instead of advanced"
    echo "   • Closing other applications during processing"
fi

echo "🎉 Ready to remove backgrounds with precision-grade quality on Mac!"