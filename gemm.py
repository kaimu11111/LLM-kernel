import torch
import numpy as np
SEED = 100
np.random.seed(SEED)
# 1. 构造测试矩阵（大小与优化版 GEMM 一致）
M, K, N = 1024, 1024, 1024
d_A = torch.randn(M, K, device='cuda', dtype=torch.float32)
d_B = torch.randn(K, N, device='cuda', dtype=torch.float32)

# 2. 预热
_ = torch.matmul(d_A, d_B)
torch.cuda.synchronize()

# 3. 多次测时
n_runs = 10
times = []
for i in range(n_runs):
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    _ = torch.matmul(d_A, d_B)
    end_evt.record()
    torch.cuda.synchronize()
    times.append(start_evt.elapsed_time(end_evt))

# 4. 打印结果
print("单次运行时间（ms）:", np.array(times))
print(f"平均时间: {np.mean(times):.3f} ms")
print(f"标准差:   {np.std(times):.3f} ms")
