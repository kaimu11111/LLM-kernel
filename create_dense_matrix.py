import cupy as cp
import cupyx.scipy.sparse as cxs
import numpy as np
from scipy.sparse import load_npz

# 加载 CSR 稀疏矩阵
csr_cpu = load_npz("sparse_matrix.npz")
csr_gpu = cxs.csr_matrix(csr_cpu)

# 创建稠密输入矩阵 X
n_cols = 128
dense_matrix = cp.random.rand(csr_gpu.shape[1], n_cols).astype(cp.float32)

# 保存为 .npz 文件（需要先转为 NumPy）
dense_matrix_np = cp.asnumpy(dense_matrix)
np.savez("dense_matrix.npz", data=dense_matrix_np)

print(f"已保存稠密矩阵：形状 = {dense_matrix_np.shape}, 文件 = dense_matrix.npz")
