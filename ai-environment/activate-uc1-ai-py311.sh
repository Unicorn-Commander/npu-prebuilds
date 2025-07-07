#!/bin/bash
# UC-1 AI Environment Activation Script (Python 3.11)
source ~/rocm_env.sh 2>/dev/null || true
source /home/ucadmin/ai-env-py311/bin/activate
echo "🦄 UC-1 AI Environment Activated (Python 3.11)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "ROCm Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Display available frameworks
echo "Available ML Frameworks:"
python -c "
import sys
frameworks = []
try:
    import torch; frameworks.append(f'PyTorch {torch.__version__}')
except: pass
try:
    import tensorflow as tf; frameworks.append(f'TensorFlow {tf.__version__}')
except: pass
try:
    import jax; frameworks.append(f'JAX {jax.__version__}')
except: pass
try:
    import onnxruntime; frameworks.append(f'ONNX Runtime {onnxruntime.__version__}')
except: pass

for fw in frameworks:
    print(f'  ✅ {fw}')
if not frameworks:
    print('  ⚠️ No frameworks detected')
"
