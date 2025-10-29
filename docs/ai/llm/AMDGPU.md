**AMD 显卡提供 GPU 加速支持**

如果需要 AMD 显卡支持，需要安装 ROCm 驱动和软件栈，AMD 显卡通过 ROCm 平台提供 GPU 加速支持。

**步骤：**
1. 安装 AMD ROCm 驱动（需 Linux 系统，如 Ubuntu 24.04）：
```sh
wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/noble/amdgpu-install_6.4.60403-1_all.deb
sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
```

2. 安装 AMD GPU 驱动
```sh
wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/noble/amdgpu-install_6.4.60403-1_all.deb
sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
```

3. 验证是否启用 GPU：
```sh
# 安装支持 ROCm 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

```python
import torch
print(torch.cuda.is_available())  # 应返回 True（ROCm 模拟 CUDA 接口）
```

4. 配置 Ollama 使用 AMD GPU 加速：
```sh
export OLLAMA_AMDGPU=1
```