import cupy as cp
import numpy as np
from scipy.sparse import load_npz, csr_matrix

# 加载稀疏矩阵
dataset = "am"
csr_A_cpu = load_npz(f"sparse_matrix_{dataset}.npz")
csr_B_cpu = load_npz(f"sparse_matrix_{dataset}.npz")  # 测试用，同一个矩阵

print(f"A shape: {csr_A_cpu.shape}, nnz: {csr_A_cpu.nnz}")
print(f"B shape: {csr_B_cpu.shape}, nnz: {csr_B_cpu.nnz}")

# 将scipy csr矩阵转换为cupy csr矩阵
def scipy_csr_to_cupy(csr):
    data = cp.asarray(csr.data)
    indices = cp.asarray(csr.indices)
    indptr = cp.asarray(csr.indptr)
    shape = csr.shape
    return cp.sparse.csr_matrix((data, indices, indptr), shape=shape)

csr_A_gpu = scipy_csr_to_cupy(csr_A_cpu)
csr_B_gpu = scipy_csr_to_cupy(csr_B_cpu)

# 调用 cuSPARSE 进行稀疏矩阵乘法
# cupy csr矩阵支持 @ 运算符直接调用 cusparseSpGEMM (cuSPARSE >= 11)
cp.cuda.Stream.null.synchronize()

n_runs = 10
total_time = 0
run_times = []

# 预热
Y = csr_A_gpu @ csr_B_gpu
cp.cuda.Stream.null.synchronize()

for i in range(n_runs):
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()

    Y = csr_A_gpu @ csr_B_gpu

    end.record()
    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end)  # 毫秒
    total_time += elapsed
    run_times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.3f} ms")

avg_time = total_time / n_runs
print(f"\nSPGEMM with cuSPARSE completed.")
print(f"Output shape: {Y.shape}")
print(f"Average time over {n_runs} runs: {avg_time:.3f} ms")
print(f"All run times: {run_times}")
print(f"Std deviation: {np.std(run_times):.3f} ms")

# 如需转回CPU可：
Y_cpu = Y.get()
from scipy.sparse import save_npz

# 保存结果矩阵 Y_cpu 到文件
save_npz(f"spgemm_result_{dataset}.npz", Y_cpu)

print(f"结果已保存为 spgemm_result_{dataset}.npz")