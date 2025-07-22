try:
    import torch
    print("✅ CUDA可用:", torch.cuda.is_available())
    print("✅ PyTorch可用,版本:", torch.__version__)
except ImportError:
    print("❌ CUDA/PyTorch:未安装/不可用")