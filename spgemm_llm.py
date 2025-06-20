import cupy as cp
import numpy as np
from scipy.sparse import load_npz

# 加载稀疏矩阵
dataset = "am"
csr_A_cpu = load_npz(f"sparse_matrix_{dataset}.npz")
csr_B_cpu = load_npz(f"sparse_matrix_{dataset}.npz")

print(f"A shape: {csr_A_cpu.shape}, nnz: {csr_A_cpu.nnz}")
print(f"B shape: {csr_B_cpu.shape}, nnz: {csr_B_cpu.nnz}")

# 转换为 CuPy 数组
def scipy_csr_to_cupy(csr):
    return (
        cp.asarray(csr.data),
        cp.asarray(csr.indices),
        cp.asarray(csr.indptr),
        csr.shape
    )

data_A, indices_A, indptr_A, (m, k) = scipy_csr_to_cupy(csr_A_cpu)
data_B, indices_B, indptr_B, (_, n) = scipy_csr_to_cupy(csr_B_cpu)

# 输出最大估计非零数
max_nnz = int(csr_A_cpu.nnz * csr_B_cpu.nnz / m)

# 分配输出缓冲区
rows_out = cp.zeros(max_nnz, dtype=cp.int32)
cols_out = cp.zeros(max_nnz, dtype=cp.int32)
vals_out = cp.zeros(max_nnz, dtype=cp.float32)
nnz_counter = cp.zeros(1, dtype=cp.int32)

# CUDA kernel                    optimized kernel
spgemm_kernel = cp.RawKernel(r'''                
extern "C" __global__
void spgemm_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    const float* __restrict__ dataA,
    const int* __restrict__ indptrB,
    const int* __restrict__ indicesB,
    const float* __restrict__ dataB,
    int* __restrict__ rows_out,
    int* __restrict__ cols_out,
    float* __restrict__ vals_out,
    int* __restrict__ nnz_counter,
    int m
) {
    const int row = blockIdx.x;
    if (row >= m) return;

    const int lane = threadIdx.x;
    const int THREADS = blockDim.x;

    __shared__ int   sh_keys[1024];
    __shared__ float sh_vals[1024];
    __shared__ int   sh_size;

    if (lane == 0) sh_size = 0;
    __syncthreads();

    int startA = indptrA[row];
    int endA   = indptrA[row + 1];

    for (int i = startA + lane; i < endA; i += THREADS) {
        int k = indicesA[i];
        float valA = dataA[i];

        int startB = indptrB[k];
        int endB   = indptrB[k + 1];

        for (int j = startB; j < endB; ++j) {
            int colB = indicesB[j];
            float valB = dataB[j];
            float val = valA * valB;

            // Simple linear probing hash insert (only works if sparse enough)
            bool inserted = false;
            for (int h = lane; h < 1024; h += THREADS) {
                int old = atomicCAS(&sh_keys[h], -1, colB);
                if (old == -1 || old == colB) {
                    atomicAdd(&sh_vals[h], val);
                    inserted = true;
                    break;
                }
            }
        }
    }

    __syncthreads();

    // Flush result from shared mem to global memory
    for (int i = lane; i < 1024; i += THREADS) {
        if (sh_keys[i] != -1) {
            int idx = atomicAdd(nnz_counter, 1);
            rows_out[idx] = row;
            cols_out[idx] = sh_keys[i];
            vals_out[idx] = sh_vals[i];
        }
    }
}

''', 'spgemm_kernel')



# original kernel
# spgemm_kernel = cp.RawKernel(r''' 
# extern "C" __global__                                                                 
# void spgemm_kernel(
#     const int* indptrA,
#     const int* indicesA,
#     const float* dataA,
#     const int* indptrB,
#     const int* indicesB,
#     const float* dataB,
#     int* rows_out,
#     int* cols_out,
#     float* vals_out,
#     int* nnz_counter,
#     int m
# ) {
#     int row = blockDim.x * blockIdx.x + threadIdx.x;
#     if (row >= m) return;

#     int startA = indptrA[row];
#     int endA = indptrA[row + 1];

#     for (int i = startA; i < endA; ++i) {
#         int k = indicesA[i];
#         float valA = dataA[i];

#         int startB = indptrB[k];
#         int endB = indptrB[k + 1];

#         for (int j = startB; j < endB; ++j) {
#             int col = indicesB[j];
#             float valB = dataB[j];

#             int idx = atomicAdd(nnz_counter, 1);
#             if (idx < 10000000) {
#                 rows_out[idx] = row;
#                 cols_out[idx] = col;
#                 vals_out[idx] = valA * valB;
#             }
#         }
#     }
# }
# ''','spgemm_kernel')
# 预热一次
threads = 128
blocks = (m + threads - 1) // threads
spgemm_kernel((blocks,), (threads,), (
    indptr_A, indices_A, data_A,
    indptr_B, indices_B, data_B,
    rows_out, cols_out, vals_out,
    nnz_counter, m
))
cp.cuda.Stream.null.synchronize()

# 正式计时
n_runs = 10
times = []
for i in range(n_runs):
    nnz_counter[0] = 0
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()

    spgemm_kernel((blocks,), (threads,), (
        indptr_A, indices_A, data_A,
        indptr_B, indices_B, data_B,
        rows_out, cols_out, vals_out,
        nnz_counter, m
    ))

    end.record()
    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end)
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.3f} ms")

print(f"\nSpGEMM kernel complete.")
print(f"Estimated nnz: {int(nnz_counter[0].item())}")
print(f"Avg time: {np.mean(times):.3f} ms")
print(f"Std dev: {np.std(times):.3f} ms")

from scipy.sparse import coo_matrix, load_npz

# 1. 从 GPU 拷贝 kernel 输出结果
nnz = int(nnz_counter[0].item())
rows_host = cp.asnumpy(rows_out[:nnz])
cols_host = cp.asnumpy(cols_out[:nnz])
vals_host = cp.asnumpy(vals_out[:nnz])

# 2. 构造自定义kernel输出的稀疏矩阵 (COO → CSR)
result_kernel = coo_matrix((vals_host, (rows_host, cols_host)), shape=(m, n)).tocsr()

# 3. 加载参考结果
result_ref = load_npz("spgemm_result_am.npz").tocsr()

# 4. 计算差异
diff = (result_kernel - result_ref).tocoo()
if diff.nnz == 0:
    print("✅ 自定义SpGEMM kernel计算结果与参考结果完全一致！")
else:
    abs_diff = np.abs(diff.data)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    print(f"⚠️ 差异检测：共有 {diff.nnz} 个不同位置")
    print(f"最大差值: {max_diff:.2e}, 平均差值: {mean_diff:.2e}")

    # 误差容忍判断
    if max_diff < 1e-4:
        print("✅ 差异在容忍范围内（可能为浮点误差）")
    else:
        print("❌ 差异超出预期，建议检查 kernel 实现")
