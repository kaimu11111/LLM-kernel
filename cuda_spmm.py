# import cupy as cp
# import cupyx.scipy.sparse as cxs
# import numpy as np
# import time
# from scipy.sparse import load_npz

# dataset = "yelp"

# csr_cpu = load_npz(f"sparse_matrix_{dataset}.npz")
# csr_gpu = cxs.csr_matrix(csr_cpu)


# n_cols = 128
# loaded = np.load(f"dense_matrix_{dataset}.npz")
# dense_matrix_np = loaded['data']
# dense_matrix_gpu = cp.asarray(dense_matrix_np)


# _ = csr_gpu.dot(dense_matrix_gpu)

# n_runs = 10
# total_time = 0.0
# run_time = []
# for _ in range(n_runs):
#     Y = cp.empty((csr_gpu.shape[0], n_cols), dtype=cp.float32)
    
#     start = cp.cuda.Event()
#     end = cp.cuda.Event()
#     start.record()
    
#     Y = csr_gpu.dot(dense_matrix_gpu)

#     end.record()
#     end.synchronize()
    
#     elapsed = cp.cuda.get_elapsed_time(start, end)  # 单位: 毫秒
#     total_time += elapsed
#     run_time.append(elapsed)

# avg_time = total_time / n_runs

# Y_np = cp.asnumpy(Y)
# np.savez("CUDA_result.npz", result=Y_np)

# print(f"SpMM (custom kernel) completed, output shape: {Y.shape}")
# print(f"Average time over {n_runs} runs: {avg_time:.3f} ms")
# print(run_time)

import torch
import numpy as np
import time
from scipy.sparse import load_npz


dataset = "yelp"
# Load sparse matrix and convert to PyTorch
csr_cpu = load_npz(f"sparse_matrix_{dataset}.npz")
print(f"Loaded sparse matrix shape: {csr_cpu.shape}")

# Convert CSR to COO format to get row and column indices
coo_cpu = csr_cpu.tocoo()

# Convert scipy sparse matrix to PyTorch sparse tensor
indices = torch.from_numpy(np.vstack([coo_cpu.row, coo_cpu.col])).long()
values = torch.from_numpy(coo_cpu.data).float()
shape = coo_cpu.shape
sparse_tensor_cpu = torch.sparse_coo_tensor(indices, values, shape)

# Move to GPU and convert to CSR format for efficiency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sparse_tensor_gpu = sparse_tensor_cpu.to(device).to_sparse_csr()

# Load dense matrix
n_cols = 128
loaded = np.load(f"dense_matrix_{dataset}.npz")
dense_matrix_np = loaded['data']
dense_matrix_gpu = torch.from_numpy(dense_matrix_np).float().to(device)

print(f"Dense matrix shape: {dense_matrix_gpu.shape}")
print(f"Using device: {device}")

# Warmup run
_ = torch.sparse.mm(sparse_tensor_gpu, dense_matrix_gpu)

# Benchmark
n_runs = 10
total_time = 0.0
run_time = []

for i in range(n_runs):
    if device.type == 'cuda':
        # CUDA timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        Y = torch.sparse.mm(sparse_tensor_gpu, dense_matrix_gpu)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)  # milliseconds
    else:
        # CPU timing
        start_time = time.perf_counter()
        Y = torch.sparse.mm(sparse_tensor_gpu, dense_matrix_gpu)
        end_time = time.perf_counter()
        elapsed = (end_time - start_time) * 1000  # convert to milliseconds
    
    total_time += elapsed
    run_time.append(elapsed)
    print(f"Run {i+1}: {elapsed:.3f} ms")

avg_time = total_time / n_runs

# Save result
Y_np = Y.cpu().numpy()
np.savez("CUDA_result.npz", result=Y_np)

print(f"\nSpMM (PyTorch sparse.mm) completed, output shape: {Y.shape}")
print(f"Average time over {n_runs} runs: {avg_time:.3f} ms")
print(f"All run times: {run_time}")
print(f"Standard deviation: {np.std(run_time):.3f} ms")