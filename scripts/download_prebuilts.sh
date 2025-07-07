#!/bin/bash
# NPU Prebuilt Binary Downloader
# Downloads pre-compiled MLIR-AIE and NPU development tools

set -e

# Configuration
BINARY_BASE_URL="${NPU_BINARY_URL:-https://your-hosting-domain.com/npu-binaries}"
INSTALL_DIR="${NPU_INSTALL_DIR:-$HOME/npu-dev}"
TEMP_DIR="/tmp/npu-install-$$"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create directories
mkdir -p "$INSTALL_DIR" "$TEMP_DIR"
cd "$TEMP_DIR"

log_info "🦄 Magic Unicorn NPU Prebuilt Downloader"
log_info "Installing to: $INSTALL_DIR"
log_info "Download source: $BINARY_BASE_URL"

# Check requirements
check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        log_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    if ! command -v tar &> /dev/null; then
        log_error "tar not found. Please install tar."
        exit 1
    fi
    
    log_success "Requirements check passed"
}

# Download function with fallback
download_file() {
    local url=$1
    local output=$2
    
    log_info "📦 Downloading $(basename "$url")..."
    
    if command -v wget &> /dev/null; then
        wget -O "$output" "$url" --progress=bar:force 2>&1 || return 1
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "$url" --progress-bar || return 1
    else
        log_error "No download tool available"
        return 1
    fi
}

# Verify checksums if available
verify_checksums() {
    local package=$1
    
    if [ -f "${package}.sha256" ]; then
        log_info "🔍 Verifying checksums..."
        if sha256sum -c "${package}.sha256" &>/dev/null; then
            log_success "Checksum verification passed"
        else
            log_error "Checksum verification failed!"
            return 1
        fi
    else
        log_warning "No checksums available for verification"
    fi
}

# Download and install package
install_package() {
    local package=$1
    local description=$2
    
    log_info "Installing $description..."
    
    # Download main package
    download_file "$BINARY_BASE_URL/${package}.tar.gz" "${package}.tar.gz" || {
        log_error "Failed to download $package"
        return 1
    }
    
    # Download checksums if available
    download_file "$BINARY_BASE_URL/${package}.sha256" "${package}.sha256" 2>/dev/null || true
    
    # Verify checksums
    verify_checksums "$package"
    
    # Extract package
    log_info "📂 Extracting $package..."
    tar -xzf "${package}.tar.gz" -C "$INSTALL_DIR/" || {
        log_error "Failed to extract $package"
        return 1
    }
    
    log_success "$description installed successfully"
}

# Main installation
main() {
    check_requirements
    
    echo
    log_info "📦 Available NPU Prebuilt Packages:"
    echo "  • MLIR-AIE Toolkit: Compiler and runtime (~80MB compressed)"
    echo "  • XRT Runtime: NPU device management (~25MB compressed)"
    echo "  • Python Bindings: Development libraries (~35MB compressed)"
    echo
    
    # Install MLIR-AIE prebuilts
    install_package "mlir-aie-prebuilts" "MLIR-AIE Toolkit"
    
    # Install additional packages if available
    if download_file "$BINARY_BASE_URL/xrt-runtime.tar.gz" "xrt-runtime.tar.gz" 2>/dev/null; then
        tar -xzf "xrt-runtime.tar.gz" -C "$INSTALL_DIR/"
        log_success "XRT Runtime installed"
    else
        log_warning "XRT Runtime package not available - will use system installation"
    fi
    
    # Set up environment
    log_info "⚙️ Setting up environment..."
    
    cat > "$INSTALL_DIR/setup_npu_env.sh" << 'EOF'
#!/bin/bash
# NPU Development Environment Setup

# Add NPU tools to PATH
export PATH="$HOME/npu-dev/mlir-aie-prebuilts/bin:$PATH"

# Set up library paths
export LD_LIBRARY_PATH="$HOME/npu-dev/mlir-aie-prebuilts/lib:$LD_LIBRARY_PATH"

# Python path for NPU bindings
export PYTHONPATH="$HOME/npu-dev/mlir-aie-prebuilts/python:$PYTHONPATH"

# XRT setup
if [ -f "/opt/xilinx/xrt/setup.sh" ]; then
    source /opt/xilinx/xrt/setup.sh
elif [ -f "$HOME/npu-dev/xrt-runtime/setup.sh" ]; then
    source "$HOME/npu-dev/xrt-runtime/setup.sh"
fi

echo "🦄 NPU Development Environment Ready"
echo "Tools available: aie-opt, aie-translate, aie-lsp-server"
echo "Python: $(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || echo 'Not configured')"

# Verify NPU availability
if command -v xrt-smi &>/dev/null; then
    echo "NPU Status: $(xrt-smi examine 2>/dev/null | grep -i phoenix || echo 'NPU not detected')"
else
    echo "NPU Status: XRT not available"
fi
EOF
    
    chmod +x "$INSTALL_DIR/setup_npu_env.sh"
    
    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"
    
    echo
    log_success "🎉 NPU prebuilt installation completed!"
    echo
    echo "🚀 To activate the NPU development environment:"
    echo "   source $INSTALL_DIR/setup_npu_env.sh"
    echo
    echo "🔧 To verify installation:"
    echo "   aie-opt --version"
    echo "   xrt-smi examine"
    echo
    log_info "Integration with Magic Unicorn TTS:"
    echo "   curl -fsSL https://raw.githubusercontent.com/Unicorn-Commander/magic-unicorn-tts/main/install.sh | bash"
}

# Error handling
trap 'log_error "Installation failed! Cleaning up..."; rm -rf "$TEMP_DIR" 2>/dev/null || true; exit 1' ERR

# Run installation
main "$@"