import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("cuDNN 版本:", torch.backends.cudnn.version())
    print("GPU 数量:", torch.cuda.device_count())
    print("GPU 名称:", torch.cuda.get_device_name(0))

    # 简单测试 GPU 运算
    a = torch.rand(1000, 1000).cuda()
    b = torch.rand(1000, 1000).cuda()
    c = torch.matmul(a, b)
    print("GPU 矩阵乘法测试通过！")
else:
    print("⚠️ CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch 安装。")