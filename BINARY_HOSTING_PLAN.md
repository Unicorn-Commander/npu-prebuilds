# 📦 **NPU Binary Hosting Strategy**

## 🎯 **Overview**

Large NPU precompiled binaries (MLIR-AIE, XRT, etc.) are stored separately for flexible hosting options. This document outlines the hosting strategy and download integration.

## 📁 **Binary Package Structure**

### **Available at**: `/home/ucadmin/Development/github_repos/npu-binaries-for-hosting/`

```
npu-binaries-for-hosting/
└── mlir-aie-prebuilts/
    ├── bin/                    # Compiled MLIR-AIE tools (150MB+)
    │   ├── aie-opt             # 120MB - MLIR optimization tool
    │   ├── aie-lsp-server      # 94MB - Language server
    │   ├── aie-translate       # MLIR translation tool
    │   └── ...
    ├── lib/                    # Runtime libraries
    │   ├── libAIE.a
    │   ├── libMLIRAIEDialect.a
    │   └── ...
    ├── python/                 # Python bindings (150MB+)
    │   └── aie/_mlir_libs/
    │       ├── libAIEAggregateCAPI.so    # 152MB
    │       └── ...
    └── include/                # Header files
        └── aie_api/
```

## 🌐 **Hosting Options**

### **Option 1: Custom Web Server**
```bash
# Host via nginx/apache
server {
    location /npu-binaries/ {
        alias /path/to/npu-binaries-for-hosting/;
        autoindex on;
    }
}

# Download URLs:
# https://your-domain.com/npu-binaries/mlir-aie-prebuilts.tar.gz
```

### **Option 2: Git LFS on Private Server**
```bash
# Initialize Git LFS repository
cd /home/ucadmin/Development/github_repos/npu-binaries-for-hosting/
git init
git lfs install
git lfs track "*.so" "*.a" "bin/*"
git add .
git commit -m "NPU precompiled binaries"

# Push to your private Git server
git remote add origin https://your-git-server.com/npu-binaries.git
git push -u origin main
```

### **Option 3: Object Storage (S3-style)**
```bash
# Upload to MinIO/S3-compatible storage
aws s3 sync mlir-aie-prebuilts/ s3://your-bucket/npu-prebuilds/mlir-aie/

# Public download URLs:
# https://your-storage.com/npu-prebuilds/mlir-aie/bin/aie-opt
```

### **Option 4: Direct Download via wget/curl**
```bash
# Simple HTTP server for direct downloads
cd /home/ucadmin/Development/github_repos/npu-binaries-for-hosting/
python3 -m http.server 8080

# Or create compressed archives
tar -czf mlir-aie-prebuilts.tar.gz mlir-aie-prebuilts/
```

## 🔧 **Integration with Installation Scripts**

### **Download Script Template**
```bash
#!/bin/bash
# install_npu_binaries.sh

BINARY_BASE_URL="https://your-hosting-domain.com/npu-binaries"
INSTALL_DIR="$HOME/npu-dev"

download_and_extract() {
    local package=$1
    local url="$BINARY_BASE_URL/$package.tar.gz"
    
    echo "📦 Downloading $package..."
    wget -O "/tmp/$package.tar.gz" "$url" || {
        echo "❌ Failed to download $package"
        return 1
    }
    
    echo "📂 Extracting $package..."
    tar -xzf "/tmp/$package.tar.gz" -C "$INSTALL_DIR/"
    
    echo "✅ $package installed successfully"
}

# Download MLIR-AIE prebuilts
download_and_extract "mlir-aie-prebuilts"

# Set up environment
export PATH="$INSTALL_DIR/mlir-aie-prebuilts/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/mlir-aie-prebuilts/lib:$LD_LIBRARY_PATH"
```

### **Integration in Main install_npu_stack.sh**
```bash
# Add to existing install script
echo "🚀 Installing NPU precompiled binaries..."

if command -v wget &> /dev/null; then
    bash <(curl -fsSL https://your-domain.com/install_npu_binaries.sh)
else
    echo "⚠️  wget not found. Please install manually:"
    echo "   wget https://your-domain.com/npu-binaries/mlir-aie-prebuilds.tar.gz"
fi
```

## 📊 **Binary Package Sizes**

| Package | Compressed | Uncompressed | Description |
|---------|------------|--------------|-------------|
| `mlir-aie-prebuilts.tar.gz` | ~80MB | ~400MB | Complete MLIR-AIE toolkit |
| `bin/` only | ~45MB | ~250MB | Essential tools only |
| `python/` only | ~35MB | ~150MB | Python bindings only |

## 🔐 **Security Considerations**

### **Checksums and Verification**
```bash
# Generate checksums
find mlir-aie-prebuilts -type f -exec sha256sum {} \; > checksums.sha256

# Verification in install script
echo "🔍 Verifying binary integrity..."
sha256sum -c checksums.sha256 || {
    echo "❌ Binary verification failed!"
    exit 1
}
```

### **Signed Downloads**
```bash
# GPG sign the archive
gpg --armor --detach-sign mlir-aie-prebuilts.tar.gz

# Verify in install script
gpg --verify mlir-aie-prebuilts.tar.gz.asc mlir-aie-prebuilts.tar.gz
```

## 📋 **Deployment Checklist**

### **Before Hosting**
- [ ] Test all binaries work on target systems
- [ ] Generate compressed archives
- [ ] Create checksums and signatures
- [ ] Test download and extraction scripts

### **Hosting Setup**
- [ ] Choose hosting method (web server/Git LFS/object storage)
- [ ] Configure public access and bandwidth limits
- [ ] Set up SSL certificates for HTTPS
- [ ] Test download speeds and reliability

### **Integration Testing**
- [ ] Update install scripts with download URLs
- [ ] Test complete installation from scratch
- [ ] Verify all tools work after download
- [ ] Document any additional requirements

## 🚀 **Quick Start for Implementation**

1. **Choose hosting method** based on your infrastructure
2. **Compress binaries**: `tar -czf mlir-aie-prebuilts.tar.gz mlir-aie-prebuilts/`
3. **Upload to hosting location**
4. **Update install scripts** with download URLs
5. **Test end-to-end installation**

---

**🦄 Binary hosting strategy for Magic Unicorn NPU ecosystem**

*Flexible deployment options for maximum accessibility*