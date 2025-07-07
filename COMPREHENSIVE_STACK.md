# 🚀 **Comprehensive NPU Development Stack**

## 📦 **What's Included**

This repository contains the complete AMD Ryzen AI NPU development ecosystem, harvested from multiple successful production projects:

### **🎯 Core NPU Development**
- **Master NPU-Development Toolkit**: Complete installation, documentation, and verification
- **MLIR-AIE Prebuilts**: Pre-compiled libraries and runtime components
- **AI Environment Setup**: Python 3.11 ML/AI environment with ROCm integration

### **🦄 Proven Implementations**
- **Magic Unicorn TTS**: NPU-optimized Kokoro TTS with 35% performance improvement
- **Whisper NPU**: Speech recognition acceleration examples
- **NPU Acceleration Frameworks**: Production-ready optimization patterns

## 📁 **Directory Structure**

```
npu-prebuilds/
├── documentation/              # Comprehensive NPU guides
│   ├── NPU_DEVELOPER_GUIDE.md
│   ├── VITIS_AI_MLIR_INTEGRATION.md
│   ├── NPU_OPTIMIZATION_GUIDE.md
│   └── NPU_USE_CASES_GUIDE.md
├── scripts/                    # Installation & verification
│   ├── install_npu_stack.sh
│   └── verify_npu_setup.sh
├── ai-environment/             # Python ML environment setup
│   ├── activate-uc1-ai.sh
│   └── activate-uc1-ai-py311.sh
├── mlir-aie-prebuilts/         # Pre-compiled MLIR-AIE libraries
│   ├── bin/                    # Compiled tools
│   ├── lib/                    # Runtime libraries
│   └── python/                 # Python bindings
├── software/                   # Software requirements
└── examples/                   # NPU application examples
```

## ⚡ **Quick Start**

### 1. **Complete NPU Stack Installation**
```bash
git clone https://github.com/Unicorn-Commander/npu-prebuilds.git
cd npu-prebuilds

# Install complete NPU development environment
sudo ./scripts/install_npu_stack.sh

# Verify installation
./scripts/verify_npu_setup.sh
```

### 2. **Activate AI Environment** 
```bash
# Activate comprehensive AI environment
source ai-environment/activate-uc1-ai-py311.sh

# Verify frameworks
python -c "import torch, onnxruntime; print('NPU-ready!')"
```

### 3. **Use Magic Unicorn TTS NPU Integration**
```bash
# Install Magic Unicorn TTS (uses these prebuilds)
curl -fsSL https://raw.githubusercontent.com/Unicorn-Commander/magic-unicorn-tts/main/install.sh | bash
```

## 🎯 **Value Proposition**

### **Time Savings**
- **2+ hours compilation time** → **5 minute installation**
- **Complete environment setup** → **Single command activation**
- **Tested configurations** → **Guaranteed compatibility**

### **Production-Ready Components**
- ✅ **MLIR-AIE**: Complete build with all dependencies
- ✅ **XRT Runtime**: NPU device management 
- ✅ **Python Environment**: ML/AI frameworks + NPU support
- ✅ **Verification Tools**: Comprehensive testing suite

### **Real-World Optimization**
- 🚀 **Magic Unicorn TTS**: 35% performance improvement patterns
- 🎤 **Whisper NPU**: Speech recognition acceleration techniques
- 📊 **Proven Benchmarks**: Measurable performance data

## 🔧 **Components Detail**

### **MLIR-AIE Prebuilts**
Pre-compiled AMD MLIR-AIE framework with:
- Phoenix NPU kernel compilation support
- Python bindings and tools
- Runtime libraries and dependencies
- Examples and documentation

### **AI Environment** 
Complete Python 3.11 environment featuring:
- **PyTorch**: With ROCm GPU support
- **ONNX Runtime**: NPU execution provider
- **TensorFlow**: Additional ML framework
- **Jupyter**: Development environment
- **Specialized packages**: Audio, vision, NLP libraries

### **NPU Development Tools**
Production-tested utilities:
- NPU detection and status tools
- Performance profiling and benchmarking
- Driver installation and verification
- Debugging and diagnostic tools

## 🌟 **Success Stories**

### **Magic Unicorn TTS**
- **Performance**: 35% speedup on NPU vs CPU
- **Implementation**: Complete TTS application with web interface
- **Models**: INT8/FP16 quantized for NPU optimization
- **Real-time Factor**: 0.26 (sub-real-time synthesis)

### **Whisper NPU** 
- **Speech Recognition**: NPU-accelerated Whisper implementation
- **Always-on Listening**: Low-power NPU voice activity detection
- **Production GUI**: Complete application with system integration

## 📋 **Hardware Requirements**

### **Supported NPU Hardware**
- ✅ **AMD Ryzen AI Phoenix** (Primary target)
- ✅ **AMD Ryzen AI Strix Point** (Compatible)
- ✅ **AMD Ryzen AI Hawk Point** (Compatible)

### **Software Requirements**
- **OS**: Ubuntu 22.04+ (Linux kernel 6.10+)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ for complete development environment

## 🚀 **Getting Started**

Ready to accelerate your NPU development? Start with:

1. **[NPU Prebuilds](https://github.com/Unicorn-Commander/npu-prebuilds)** - This repository
2. **[Magic Unicorn TTS](https://github.com/Unicorn-Commander/magic-unicorn-tts)** - Complete TTS application
3. **[AMD NPU Utils](https://github.com/Unicorn-Commander/amd-npu-utils)** - Development tools

---

**🦄 Developed by Magic Unicorn Unconventional Technology & Stuff Inc**

*Turning NPU development from hours to minutes*