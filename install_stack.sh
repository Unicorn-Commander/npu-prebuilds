#!/bin/bash
set -e

# NPU Prebuilds - Complete Stack Installer
# Installs all pre-compiled NPU components

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

INSTALL_DIR="/opt/npu-stack"
TEMP_DIR="/tmp/npu-prebuilds"
PREBUILDS_URL="https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download"

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
    🚀 NPU Prebuilds Installer 🚀
    Pre-compiled AMD Ryzen AI Components
    
    Components:
    • MLIR-AIE Runtime
    • VitisAI ONNX Runtime  
    • Quantized Models
    • Development Tools
EOF
    echo -e "${NC}"
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check for NPU hardware
    if lsmod | grep -q "amdxdna"; then
        success "AMD NPU driver detected"
        NPU_AVAILABLE=true
    else
        warn "AMD NPU driver not detected"
        NPU_AVAILABLE=false
    fi
    
    # Check sudo access
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root - this is not recommended"
    else
        # Test sudo access
        if ! sudo -n true 2>/dev/null; then
            log "This installer requires sudo access for system installation"
            sudo -v || error "Sudo access required"
        fi
    fi
    
    # Check available space
    AVAILABLE_SPACE=$(df /opt 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
    REQUIRED_SPACE=5000000  # 5GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        warn "Low disk space. Available: ${AVAILABLE_SPACE}KB, Required: ${REQUIRED_SPACE}KB"
    fi
}

setup_directories() {
    log "Setting up installation directories..."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Create install directory
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown $(whoami):$(id -gn) "$INSTALL_DIR"
    
    success "Directories created"
}

download_components() {
    log "Downloading NPU prebuilds..."
    
    # List of components to download
    COMPONENTS=(
        "mlir-aie-runtime.tar.gz"
        "onnxruntime-vitisai.whl"
        "kokoro-npu-models.tar.gz"
        "npu-dev-tools.tar.gz"
    )
    
    for component in "${COMPONENTS[@]}"; do
        log "Downloading $component..."
        if curl -fsSL -o "$component" "$PREBUILDS_URL/$component"; then
            success "Downloaded $component"
        else
            warn "Failed to download $component - may not be available yet"
        fi
    done
}

install_mlir_aie() {
    if [[ -f "mlir-aie-runtime.tar.gz" ]]; then
        log "Installing MLIR-AIE runtime..."
        
        tar -xzf mlir-aie-runtime.tar.gz -C "$INSTALL_DIR"
        
        # Set up environment
        cat > "$INSTALL_DIR/setup_mlir_aie.sh" << 'EOF'
#!/bin/bash
export MLIR_AIE_PATH="/opt/npu-stack/mlir-aie"
export PATH="$MLIR_AIE_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_AIE_PATH/lib:$LD_LIBRARY_PATH"
EOF
        
        chmod +x "$INSTALL_DIR/setup_mlir_aie.sh"
        success "MLIR-AIE runtime installed"
    else
        warn "MLIR-AIE runtime not available for download"
    fi
}

install_vitisai() {
    if [[ -f "onnxruntime-vitisai.whl" ]]; then
        log "Installing VitisAI ONNX Runtime..."
        
        # Install globally or in virtual environment
        if [[ -n "$VIRTUAL_ENV" ]]; then
            pip install onnxruntime-vitisai.whl
            success "VitisAI ONNX Runtime installed in virtual environment"
        else
            sudo pip3 install onnxruntime-vitisai.whl
            success "VitisAI ONNX Runtime installed globally"
        fi
    else
        warn "VitisAI ONNX Runtime not available for download"
    fi
}

install_models() {
    if [[ -f "kokoro-npu-models.tar.gz" ]]; then
        log "Installing quantized models..."
        
        mkdir -p "$INSTALL_DIR/models"
        tar -xzf kokoro-npu-models.tar.gz -C "$INSTALL_DIR/models"
        
        success "Quantized models installed"
    else
        warn "Quantized models not available for download"
    fi
}

install_dev_tools() {
    if [[ -f "npu-dev-tools.tar.gz" ]]; then
        log "Installing development tools..."
        
        tar -xzf npu-dev-tools.tar.gz -C "$INSTALL_DIR"
        
        # Add tools to PATH
        sudo ln -sf "$INSTALL_DIR/dev-tools/bin/"* /usr/local/bin/ 2>/dev/null || true
        
        success "Development tools installed"
    else
        warn "Development tools not available for download"
    fi
}

setup_environment() {
    log "Setting up environment..."
    
    # Create global environment setup
    cat > "$INSTALL_DIR/setup_npu_env.sh" << 'EOF'
#!/bin/bash
# NPU Prebuilds Environment Setup

export NPU_STACK_PATH="/opt/npu-stack"
export MLIR_AIE_PATH="$NPU_STACK_PATH/mlir-aie"
export VITISAI_PATH="$NPU_STACK_PATH/vitisai"
export NPU_MODELS_PATH="$NPU_STACK_PATH/models"

export PATH="$MLIR_AIE_PATH/bin:$VITISAI_PATH/bin:$NPU_STACK_PATH/dev-tools/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_AIE_PATH/lib:$VITISAI_PATH/lib:$LD_LIBRARY_PATH"

echo "🚀 NPU Stack environment loaded"
echo "MLIR-AIE: $MLIR_AIE_PATH"
echo "VitisAI: $VITISAI_PATH"
echo "Models: $NPU_MODELS_PATH"
EOF
    
    chmod +x "$INSTALL_DIR/setup_npu_env.sh"
    
    # Add to system profile
    cat > /tmp/npu-stack.sh << EOF
# NPU Stack Environment
if [[ -f "$INSTALL_DIR/setup_npu_env.sh" ]]; then
    source "$INSTALL_DIR/setup_npu_env.sh"
fi
EOF
    
    sudo mv /tmp/npu-stack.sh /etc/profile.d/
    
    success "Environment setup complete"
}

verify_installation() {
    log "Verifying installation..."
    
    # Source environment
    source "$INSTALL_DIR/setup_npu_env.sh"
    
    # Check MLIR-AIE
    if [[ -f "$MLIR_AIE_PATH/bin/aie-opt" ]]; then
        success "MLIR-AIE tools found"
    else
        warn "MLIR-AIE tools not found"
    fi
    
    # Check models
    if [[ -d "$NPU_MODELS_PATH" ]] && [[ -n "$(ls -A "$NPU_MODELS_PATH" 2>/dev/null)" ]]; then
        success "NPU models found"
    else
        warn "NPU models not found"
    fi
    
    # Check Python packages
    if python3 -c "import onnxruntime; print('VitisAI provider available:', 'VitisAIExecutionProvider' in onnxruntime.get_available_providers())" 2>/dev/null; then
        success "VitisAI ONNX Runtime working"
    else
        warn "VitisAI ONNX Runtime not working"
    fi
}

cleanup() {
    log "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    success "Cleanup complete"
}

print_completion() {
    echo -e "${GREEN}"
    cat << "EOF"
    
    🎉 NPU Prebuilds Installation Complete! 🎉
    
    🚀 What's Installed:
    • MLIR-AIE Runtime - NPU kernel compilation
    • VitisAI ONNX Runtime - Quantized inference
    • Optimized Models - Ready-to-use NPU models
    • Development Tools - NPU debugging utilities
    
    📁 Installation Directory: /opt/npu-stack
    
    🔧 Usage:
    
    Load environment:
    source /opt/npu-stack/setup_npu_env.sh
    
    Or automatically on login (already configured):
    # Environment loaded via /etc/profile.d/npu-stack.sh
    
    🧪 Test Installation:
    python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
    
    📚 Documentation:
    https://github.com/Unicorn-Commander/npu-prebuilds
    
EOF
    echo -e "${NC}"
    
    if [[ "$NPU_AVAILABLE" == true ]]; then
        success "NPU hardware detected - ready for acceleration!"
    else
        warn "NPU hardware not detected - install AMD NPU drivers"
        log "See: https://github.com/Unicorn-Commander/amd-npu-utils"
    fi
}

# Main installation flow
main() {
    print_banner
    
    check_requirements
    setup_directories
    download_components
    install_mlir_aie
    install_vitisai
    install_models
    install_dev_tools
    setup_environment
    verify_installation
    cleanup
    
    print_completion
}

# Check if running as part of another script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi