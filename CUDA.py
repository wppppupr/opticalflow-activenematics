import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print(f"CUDA capability: {torch.cuda.is_available()}")
    # デバッグ情報の表示
    import torch.utils.cpp_extension
    print(f"CUDA Home: {torch.utils.cpp_extension.CUDA_HOME}")