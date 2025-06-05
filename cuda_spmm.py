import cupy as cp
import cupyx.scipy.sparse as cxs
import numpy as np
import time
from scipy.sparse import load_npz


csr_cpu = load_npz("sparse_matrix.npz")
csr_gpu = cxs.csr_matrix(csr_cpu)


n_cols = 128
loaded = np.load("dense_matrix.npz")
dense_matrix_np = loaded['data']
dense_matrix_gpu = cp.asarray(dense_matrix_np)


_ = csr_gpu.dot(dense_matrix_gpu)

n_runs = 10
total_time = 0.0
run_time = []
for _ in range(n_runs):
    Y = cp.empty((csr_gpu.shape[0], n_cols), dtype=cp.float32)
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    
    Y = csr_gpu.dot(dense_matrix_gpu)

    end.record()
    end.synchronize()
    
    elapsed = cp.cuda.get_elapsed_time(start, end)  # 单位: 毫秒
    total_time += elapsed
    run_time.append(elapsed)

avg_time = total_time / n_runs

Y_np = cp.asnumpy(Y)
np.savez("CUDA_result.npz", result=Y_np)

print(f"SpMM (custom kernel) completed, output shape: {Y.shape}")
print(f"Average time over {n_runs} runs: {avg_time:.3f} ms")
print(run_time)
