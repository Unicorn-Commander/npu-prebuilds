# 🚀 NPU Prebuilds

**Pre-compiled Components for AMD Ryzen AI NPU Development**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NPU Ready](https://img.shields.io/badge/NPU-Ready-blue.svg)](https://github.com/Unicorn-Commander/npu-prebuilds)
[![Build Time Saved](https://img.shields.io/badge/Build%20Time-2%2B%20Hours%20Saved-green.svg)](https://github.com/Unicorn-Commander/npu-prebuilds)

> ⚡ **Skip the 2+ hour compilation process**  
> Pre-built MLIR-AIE, VitisAI, and optimized models ready for immediate use

## 📦 Available Prebuilds

### Core NPU Components
- **MLIR-AIE Runtime** - NPU kernel execution environment
- **VitisAI ONNX Runtime** - Quantized model inference
- **AMD NPU Drivers** - Kernel modules and userspace tools
- **XRT Runtime** - Xilinx Runtime for NPU communication

### Optimized Models
- **Kokoro TTS Models** - INT8/FP16 quantized for NPU
- **Whisper Models** - NPU-optimized speech recognition
- **Custom Kernels** - Hand-tuned NPU operations

### Python Wheels
- **onnxruntime-vitisai** - VitisAI-enabled ONNX Runtime
- **torch-npu** - PyTorch NPU backend
- **ml-frameworks** - Pre-compiled ML dependencies

## 🎯 Supported Hardware

- **AMD Ryzen AI (Phoenix)** - Ryzen 7040/8040 series
- **AMD Ryzen AI (Strix Point)** - Ryzen AI 300 series
- **Future NPU Architectures** - Forward compatibility

## 📥 Quick Download

### Individual Components
```bash
# MLIR-AIE Runtime (Latest)
curl -fsSL -o mlir-aie-runtime.tar.gz \
  https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/mlir-aie-runtime.tar.gz

# VitisAI ONNX Runtime
curl -fsSL -o onnxruntime-vitisai.whl \
  https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/onnxruntime-vitisai.whl

# Quantized Kokoro Models
curl -fsSL -o kokoro-npu-models.tar.gz \
  https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/kokoro-npu-models.tar.gz
```

### Complete NPU Stack
```bash
curl -fsSL https://raw.githubusercontent.com/Unicorn-Commander/npu-prebuilds/main/install_stack.sh | bash
```

## 🔧 Installation

### Method 1: Automated Installer
```bash
wget https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/install_npu_stack.sh
chmod +x install_npu_stack.sh
./install_npu_stack.sh
```

### Method 2: Manual Installation
```bash
# Download and extract MLIR-AIE
wget https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/mlir-aie-runtime.tar.gz
tar -xzf mlir-aie-runtime.tar.gz -C /opt/

# Install VitisAI ONNX Runtime
pip install https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/onnxruntime-vitisai.whl

# Extract quantized models
wget https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/kokoro-npu-models.tar.gz
tar -xzf kokoro-npu-models.tar.gz
```

## 📊 Build Information

### Compilation Details
- **Compiler**: Clang 15+ with NPU extensions
- **Optimization**: -O3 with NPU-specific flags
- **Target**: AMD XDNA architecture
- **Build System**: CMake 3.20+ with custom toolchain

### Performance Benchmarks
| Component | Build Time Saved | Performance | Size |
|-----------|------------------|-------------|------|
| MLIR-AIE | 45+ minutes | Native NPU | 850MB |
| VitisAI RT | 30+ minutes | Quantized | 120MB |
| Models | 15+ minutes | Optimized | 2.1GB |
| **Total** | **90+ minutes** | **Full NPU** | **3.1GB** |

## 🏗️ Available Releases

### Latest Release: v1.2.0
- MLIR-AIE 2024.2 with NPU optimizations
- VitisAI 3.5 with INT8/FP16 support
- Kokoro TTS models (5 voices, quantized)
- PyTorch 2.1+ with NPU backend

### Previous Releases
- **v1.1.0** - Initial Kokoro TTS support
- **v1.0.0** - Basic MLIR-AIE and VitisAI

## 🔧 Usage in Projects

### Magic Unicorn TTS
```bash
# Automatically downloads prebuilds
curl -fsSL https://raw.githubusercontent.com/Unicorn-Commander/magic-unicorn-tts/main/install.sh | bash
```

### Custom Projects
```python
# Python usage example
import subprocess
import os

# Download and extract MLIR-AIE
subprocess.run([
    'curl', '-fsSL', '-o', 'mlir-aie.tar.gz',
    'https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/mlir-aie-runtime.tar.gz'
])
subprocess.run(['tar', '-xzf', 'mlir-aie.tar.gz'])

# Set environment
os.environ['MLIR_AIE_PATH'] = './mlir-aie'
os.environ['VITIS_AI_PATH'] = './vitisai'
```

## 🧪 Verification

### Test NPU Stack
```bash
# Download verification script
curl -fsSL -o test_npu_stack.py \
  https://raw.githubusercontent.com/Unicorn-Commander/npu-prebuilds/main/scripts/test_npu_stack.py

python test_npu_stack.py
```

### Expected Output
```
✅ MLIR-AIE runtime: OK
✅ VitisAI provider: OK  
✅ NPU device: Detected (Phoenix)
✅ Quantized models: 5 voices loaded
🚀 NPU stack ready for inference!
```

## 📁 Repository Structure

```
npu-prebuilds/
├── releases/
│   ├── v1.2.0/
│   │   ├── mlir-aie-runtime.tar.gz
│   │   ├── onnxruntime-vitisai.whl
│   │   ├── kokoro-npu-models.tar.gz
│   │   └── install_npu_stack.sh
│   └── latest/
├── scripts/
│   ├── build_mlir_aie.sh
│   ├── build_vitisai.sh
│   ├── test_npu_stack.py
│   └── package_models.sh
├── docs/
│   ├── BUILD.md
│   ├── USAGE.md
│   └── TROUBLESHOOTING.md
└── ci/
    ├── build_pipeline.yml
    └── test_matrix.yml
```

## 🏗️ Building from Source

If you prefer to build components yourself:

```bash
# Clone this repository
git clone https://github.com/Unicorn-Commander/npu-prebuilds.git
cd npu-prebuilds

# Build all components
./scripts/build_all.sh

# Package for distribution
./scripts/package_release.sh
```

## 🎯 Integration Examples

### CMake Integration
```cmake
# Download prebuilds in CMake
include(FetchContent)

FetchContent_Declare(
  npu_prebuilds
  URL https://github.com/Unicorn-Commander/npu-prebuilds/releases/latest/download/mlir-aie-runtime.tar.gz
)
FetchContent_MakeAvailable(npu_prebuilds)

target_link_libraries(my_app ${npu_prebuilds_SOURCE_DIR}/lib/libmlir_aie.so)
```

### Docker Integration
```dockerfile
FROM ubuntu:22.04

# Install NPU stack
RUN curl -fsSL https://raw.githubusercontent.com/Unicorn-Commander/npu-prebuilds/main/install_stack.sh | bash

# Your application setup
COPY . /app
WORKDIR /app
```

## 🤝 Contributing

### Adding New Prebuilds
1. Fork this repository
2. Add build scripts to `scripts/`
3. Update CI to build your component
4. Submit pull request with documentation

### Supported Contributions
- New ML model optimizations
- Additional NPU hardware support
- Performance improvements
- Build system enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Magic Unicorn TTS](https://github.com/Unicorn-Commander/magic-unicorn-tts) - Main TTS application
- [AMD NPU Utils](https://github.com/Unicorn-Commander/amd-npu-utils) - NPU development tools
- [MLIR-AIE](https://github.com/Xilinx/mlir-aie) - Upstream MLIR-AIE project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Unicorn-Commander/npu-prebuilds/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Unicorn-Commander/npu-prebuilds/discussions)
- **Documentation**: [Project Wiki](https://github.com/Unicorn-Commander/npu-prebuilds/wiki)

---

<div align="center">
  <p>
    <strong>Powered by Unicorn Commander 🦄</strong><br>
    <em>Accelerating NPU development for everyone</em>
  </p>
  <p>
    <a href="https://unicorncommander.com">Unicorn Commander</a> • 
    <a href="https://magicunicorn.tech">Magic Unicorn Tech</a>
  </p>
</div>