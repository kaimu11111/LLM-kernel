import cupy as cp
import cupyx.scipy.sparse as cxs
import numpy as np
from scipy.sparse import load_npz


dataset = "yelp"
# 加载 CSR 稀疏矩阵
csr_cpu = load_npz(f"sparse_matrix_{dataset}.npz")
csr_gpu = cxs.csr_matrix(csr_cpu)

# 创建稠密输入矩阵 X 和输出矩阵 Y
n_cols = 128
loaded = np.load(f"dense_matrix_{dataset}.npz")
dense_matrix_np = loaded['data']  # 获取保存时用的键名
dense_matrix_gpu = cp.asarray(dense_matrix_np)

Y = cp.zeros((csr_gpu.shape[0], n_cols), dtype=cp.float32)

# basic version              1727ms
# spmm_kernel_code = r'''
# extern "C" __global__
# void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* C,
#     int M,
#     int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;

#     for (int col = 0; col < K; ++col) {
#         float sum = 0.0f;
#         for (int jj = indptr[row]; jj < indptr[row + 1]; ++jj) {
#             int col_idx = indices[jj];
#             float val = data[jj];
#             sum += val * B[col_idx * K + col];
#         }
#         C[row * K + col] = sum;
#     }
# }
# '''

# Version 1: Basic optimizations           1107.535ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Process multiple columns per thread using unrolling
#     for (int col = 0; col < K; col += 4) {
#         float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
#         for (int jj = row_start; jj < row_end; ++jj) {
#             int col_idx = indices[jj];
#             float val = data[jj];
            
#             // Unroll the inner loop
#             if (col < K) sum0 += val * B[col_idx * K + col];
#             if (col + 1 < K) sum1 += val * B[col_idx * K + col + 1];
#             if (col + 2 < K) sum2 += val * B[col_idx * K + col + 2];
#             if (col + 3 < K) sum3 += val * B[col_idx * K + col + 3];
#         }
        
#         if (col < K) C[row * K + col] = sum0;
#         if (col + 1 < K) C[row * K + col + 1] = sum1;
#         if (col + 2 < K) C[row * K + col + 2] = sum2;
#         if (col + 3 < K) C[row * K + col + 3] = sum3;
#     }
# }
# '''

# Version 2: Using shared memory for B matrix tiles           852ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     const int TILE_SIZE = 32;
#     __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     int tid = threadIdx.x;
    
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Process B matrix in tiles
#     for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
#         // Load B tile to shared memory
#         for (int i = 0; i < TILE_SIZE && tile_start + i < K; i++) {
#             if (tid < TILE_SIZE) {
#                 B_tile[tid][i] = (tile_start + i < K) ? B[tid * K + tile_start + i] : 0.0f;
#             }
#         }
#         __syncthreads();
        
#         // Compute for this tile
#         for (int col = 0; col < TILE_SIZE && tile_start + col < K; col++) {
#             float sum = 0.0f;
#             for (int jj = row_start; jj < row_end; ++jj) {
#                 int col_idx = indices[jj];
#                 if (col_idx < TILE_SIZE) {
#                     sum += data[jj] * B_tile[col_idx][col];
#                 }
#             }
#             C[row * K + tile_start + col] = sum;
#         }
#         __syncthreads();
#     }
# }

# '''

# Version 2 optimization Version 1(Fixed shared memory approach)   1766ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     const int TILE_K = 32;
#     __shared__ float B_tile[TILE_K];  // 1D tile for column-wise access
    
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     int tid = threadIdx.x;
#     int warp_id = tid / 32;
#     int lane_id = tid % 32;
    
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Process B matrix columns in tiles
#     for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
#         int tile_end = min(k_tile + TILE_K, K);
        
#         // Each thread processes multiple columns to reduce work
#         for (int k_col = k_tile; k_col < tile_end; ++k_col) {
#             float sum = 0.0f;
            
#             // Process sparse row elements
#             for (int jj = row_start; jj < row_end; ++jj) {
#                 int sparse_col = indices[jj];
#                 float sparse_val = data[jj];
#                 sum += sparse_val * B[sparse_col * K + k_col];
#             }
            
#             C[row * K + k_col] = sum;
#         }
#     }
# }
# '''

#Version 2 optimization Version 2: Better shared memory utilization    1997ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     const int TILE_SIZE = 32;
#     __shared__ float B_shared[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     int tid = threadIdx.x;
    
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Process in tiles of the dense matrix B
#     for (int col_tile = 0; col_tile < K; col_tile += TILE_SIZE) {
#         int tile_cols = min(TILE_SIZE, K - col_tile);
        
#         // Cooperatively load B tile - each thread loads one element per iteration
#         for (int load_iter = 0; load_iter < TILE_SIZE; load_iter++) {
#             for (int col_offset = tid; col_offset < tile_cols; col_offset += blockDim.x) {
#                 int global_col = col_tile + col_offset;
#                 if (load_iter < M && global_col < K) {  // Ensure we don't go out of bounds
#                     B_shared[load_iter][col_offset] = B[load_iter * K + global_col];
#                 } else {
#                     B_shared[load_iter][col_offset] = 0.0f;
#                 }
#             }
#         }
#         __syncthreads();
        
#         // Compute for this tile
#         for (int col_offset = 0; col_offset < tile_cols; ++col_offset) {
#             float sum = 0.0f;
            
#             for (int jj = row_start; jj < row_end; ++jj) {
#                 int sparse_col = indices[jj];
#                 if (sparse_col < TILE_SIZE) {  // Only if the sparse column is in our tile
#                     sum += data[jj] * B_shared[sparse_col][col_offset];
#                 }
#             }
            
#             C[row * K + col_tile + col_offset] = sum;
#         }
#         __syncthreads();
#     }
# }
# '''


# Version 2 optimization Version 3: Optimized without shared memory      1109 ms 
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Simple but effective: process multiple columns with unrolling
#     for (int col = 0; col < K; col += 4) {
#         float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
#         // Process all sparse elements for this row
#         for (int jj = row_start; jj < row_end; ++jj) {
#             int sparse_col = indices[jj];
#             float sparse_val = data[jj];
            
#             // Unroll 4 dense columns
#             if (col < K)     sum0 += sparse_val * B[sparse_col * K + col];
#             if (col + 1 < K) sum1 += sparse_val * B[sparse_col * K + col + 1];
#             if (col + 2 < K) sum2 += sparse_val * B[sparse_col * K + col + 2];
#             if (col + 3 < K) sum3 += sparse_val * B[sparse_col * K + col + 3];
#         }
        
#         // Store results
#         if (col < K)     C[row * K + col]     = sum0;
#         if (col + 1 < K) C[row * K + col + 1] = sum1;
#         if (col + 2 < K) C[row * K + col + 2] = sum2;
#         if (col + 3 < K) C[row * K + col + 3] = sum3;
#     }
# }
# '''


# Version 2-3-1                               967 ms
# spmm_kernel_code = r'''

# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices, 
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M, int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
#     int nnz_per_row = row_end - row_start;
    
#     // Early exit for empty rows
#     if (nnz_per_row == 0) {
#         for (int col = 0; col < K; col += 4) {
#             if (col < K) C[row * K + col] = 0.0f;
#             if (col + 1 < K) C[row * K + col + 1] = 0.0f;
#             if (col + 2 < K) C[row * K + col + 2] = 0.0f;
#             if (col + 3 < K) C[row * K + col + 3] = 0.0f;
#         }
#         return;
#     }
    
#     // Process columns in chunks of 8 for better vectorization
#     for (int col_base = 0; col_base < K; col_base += 8) {
#         // Use registers for accumulation
#         float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
#         // Process sparse elements for current row
#         for (int jj = row_start; jj < row_end; ++jj) {
#             int sparse_col = indices[jj];
#             float sparse_val = data[jj];
            
#             // Calculate base address for B matrix access
#             const float* B_row = &B[sparse_col * K + col_base];
            
#             // Unroll 8 columns with explicit vectorization hints
#             #pragma unroll 8
#             for (int i = 0; i < 8; ++i) {
#                 if (col_base + i < K) {
#                     sum[i] += sparse_val * B_row[i];
#                 }
#             }
#         }
        
#         // Store results with coalesced memory access
#         float* C_row = &C[row * K + col_base];
#         #pragma unroll 8
#         for (int i = 0; i < 8; ++i) {
#             if (col_base + i < K) {
#                 C_row[i] = sum[i];
#             }
#         }
#     }
# }

# '''
####################################################################################################
# Verision 2-3-2                    59.179 ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data, 
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M, int K
# ) {
#     const int TILE_SIZE = 32;
#     const int COLS_PER_BLOCK = 8;
    
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     int col_block = blockIdx.y;
#     int col_start = col_block * COLS_PER_BLOCK;
#     int col_end = min(col_start + COLS_PER_BLOCK, K);
    
#     // Shared memory for B matrix tiles
#     __shared__ float B_shared[TILE_SIZE][COLS_PER_BLOCK];
    
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     float sum[COLS_PER_BLOCK];
#     #pragma unroll
#     for (int i = 0; i < COLS_PER_BLOCK; ++i) {
#         sum[i] = 0.0f;
#     }
    
#     // Process sparse elements in tiles
#     for (int tile_start = 0; tile_start < row_end - row_start; tile_start += TILE_SIZE) {
#         int tile_end = min(tile_start + TILE_SIZE, row_end - row_start);
        
#         // Cooperatively load B matrix data into shared memory
#         for (int jj = row_start + tile_start + threadIdx.x; 
#              jj < row_start + tile_end; 
#              jj += blockDim.x) {
#             if (jj < row_end) {
#                 int sparse_col = indices[jj];
#                 int local_idx = jj - (row_start + tile_start);
                
#                 #pragma unroll
#                 for (int c = 0; c < COLS_PER_BLOCK && col_start + c < K; ++c) {
#                     if (local_idx < TILE_SIZE) {
#                         B_shared[local_idx][c] = B[sparse_col * K + col_start + c];
#                     }
#                 }
#             }
#         }
        
#         __syncthreads();
        
#         // Compute using shared memory
#         for (int jj = row_start + tile_start; jj < row_start + tile_end; ++jj) {
#             float sparse_val = data[jj];
#             int local_idx = jj - (row_start + tile_start);
            
#             #pragma unroll
#             for (int c = 0; c < COLS_PER_BLOCK && col_start + c < K; ++c) {
#                 sum[c] += sparse_val * B_shared[local_idx][c];
#             }
#         }
        
#         __syncthreads();
#     }
    
#     // Store results
#     #pragma unroll
#     for (int c = 0; c < COLS_PER_BLOCK && col_start + c < K; ++c) {
#         C[row * K + col_start + c] = sum[c];
#     }
# }
# '''
######################################################################################################

# Test to optimize the 2-3-2
spmm_kernel_code = r'''
extern "C" __global__ void spmm_csr(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data, 
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K
) {
    const int WARP_SIZE = 32;
    const int COLS_PER_WARP = 32;
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = warp_id;
    int col = blockIdx.y * COLS_PER_WARP + lane_id;
    
    if (row >= M || col >= K) return;
    
    int row_start = indptr[row];
    int row_end = indptr[row + 1];
    
    float sum = 0.0f;
    
    // Each thread in warp handles one column
    for (int jj = row_start; jj < row_end; ++jj) {
        int sparse_col = indices[jj];
        float sparse_val = data[jj];
        sum = fmaf(sparse_val, B[sparse_col * K + col], sum);
    }
    
    C[row * K + col] = sum;
}
'''








#######################################################################################################

#Version 2-3-3                                              1050ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M, int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;
    
#     // Ensure K is multiple of 4 for vectorization
#     if (K % 4 != 0) {
#         // Fallback to scalar version for non-aligned K
#         return;
#     }
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Process 4 columns at a time using float4
#     for (int col = 0; col < K; col += 4) {
#         float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
#         for (int jj = row_start; jj < row_end; ++jj) {
#             int sparse_col = indices[jj];
#             float sparse_val = data[jj];
            
#             // Load 4 elements from B using vectorized load
#             float4 b_vals = reinterpret_cast<const float4*>(&B[sparse_col * K + col])[0];
            
#             // Vectorized multiply-accumulate
#             sum.x += sparse_val * b_vals.x;
#             sum.y += sparse_val * b_vals.y;
#             sum.z += sparse_val * b_vals.z;
#             sum.w += sparse_val * b_vals.w;
#         }
        
#         // Store using vectorized write
#         reinterpret_cast<float4*>(&C[row * K + col])[0] = sum;
#     }
# }
# '''

# Version 2 optimization Version 4: Advanced optimization with warp-level cooperation   2307ms  
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices,
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
#     int nnz_per_row = row_end - row_start;
    
#     // Adaptive strategy based on sparsity
#     if (nnz_per_row > 16) {
#         // For dense rows: process multiple columns together
#         for (int col = 0; col < K; col += 8) {
#             float sum[8] = {0.0f};
            
#             for (int jj = row_start; jj < row_end; ++jj) {
#                 int sparse_col = indices[jj];
#                 float sparse_val = data[jj];
                
#                 #pragma unroll
#                 for (int i = 0; i < 8 && col + i < K; i++) {
#                     sum[i] += sparse_val * B[sparse_col * K + col + i];
#                 }
#             }
            
#             #pragma unroll
#             for (int i = 0; i < 8 && col + i < K; i++) {
#                 C[row * K + col + i] = sum[i];
#             }
#         }
#     } else {
#         // For sparse rows: simple processing
#         for (int col = 0; col < K; ++col) {
#             float sum = 0.0f;
#             for (int jj = row_start; jj < row_end; ++jj) {
#                 sum += data[jj] * B[indices[jj] * K + col];
#             }
#             C[row * K + col] = sum;
#         }
#     }
# }
# '''
# Version 3             2915ms
# spmm_kernel_code = r'''
# extern "C" __global__ void spmm_csr(
#     const int* __restrict__ indptr,
#     const int* __restrict__ indices, 
#     const float* __restrict__ data,
#     const float* __restrict__ B,
#     float* __restrict__ C,
#     int M,
#     int K
# ) {
#     int row = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row >= M) return;
    
#     int row_start = indptr[row];
#     int row_end = indptr[row + 1];
    
#     // Simple unrolling without alignment checks
#     for (int col = 0; col < K; col += 2) {
#         float sum0 = 0.0f, sum1 = 0.0f;
        
#         for (int jj = row_start; jj < row_end; ++jj) {
#             int col_idx = indices[jj];
#             float val = data[jj];
            
#             // No alignment checks - just unroll
#             sum0 += val * B[col_idx * K + col];
#             if (col + 1 < K) {
#                 sum1 += val * B[col_idx * K + col + 1];
#             }
#         }
        
#         C[row * K + col] = sum0;
#         if (col + 1 < K) {
#             C[row * K + col + 1] = sum1;
#         }
#     }
# }
# '''

# 编译核函数
mod = cp.RawModule(code=spmm_kernel_code, backend='nvrtc')
spmm_kernel = mod.get_function('spmm_csr')

# CUDA 配置
threads_per_block = 128
num_rows = csr_gpu.shape[0]
blocks_per_grid = (num_rows + threads_per_block - 1) // threads_per_block

# 🔥 预热一次
spmm_kernel(
    (blocks_per_grid,), (threads_per_block,),
    (
        csr_gpu.indptr.astype(cp.int32),
        csr_gpu.indices.astype(cp.int32),
        csr_gpu.data,
        dense_matrix_gpu,
        Y,
        cp.int32(num_rows),
        cp.int32(n_cols)
    )
)
cp.cuda.Device(0).synchronize()

# ⏱ 10 次运行计时
n_runs = 10
total_time = 0.0
run_time = []
for _ in range(n_runs):
    Y.fill(0)  # 清空输出
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()

    spmm_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (
            csr_gpu.indptr.astype(cp.int32),
            csr_gpu.indices.astype(cp.int32),
            csr_gpu.data,
            dense_matrix_gpu,
            Y,
            cp.int32(num_rows),
            cp.int32(n_cols)
        )
    )

    end.record()
    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end)
    total_time += elapsed
    run_time.append(elapsed)
avg_time = total_time / n_runs
print(f"SpMM (custom kernel) completed, output shape: {Y.shape}")
print(f"Average time over {n_runs} runs: {avg_time:.3f} ms")
print(run_time)


loaded = np.load("CUDA_result.npz")
CUDA_Y = loaded["result"]
CUDA_Y_cp = cp.asarray(CUDA_Y)
abs_error = np.abs(CUDA_Y_cp - Y)
max_abs_err = np.max(abs_error)
mean_abs_err = np.mean(abs_error)
print(f"Max error: {max_abs_err:.6e}")
print(f"abs mean error: {mean_abs_err:.6e}")