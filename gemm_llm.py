import cupy as cp
import numpy as np
import torch
SEED = 100
np.random.seed(SEED)
# -----------------------
# 1. 生成或加载测试矩阵
# -----------------------
M, K, N = 1024, 1024, 1024
h_A = np.random.randn(M, K).astype(np.float32)
h_B = np.random.randn(K, N).astype(np.float32)

# 拷到 GPU
d_A = cp.asarray(h_A)
d_B = cp.asarray(h_B)
d_C = cp.zeros((M, N), dtype=cp.float32)

# -----------------------
# 2. 定义 CUDA Kernel
# -----------------------
# kernel_code = r'''
# #define TILE 16
# #define WARP_SIZE 32

# __inline__ __device__
# float warpReduceSum(float val) {
#     for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#         val += __shfl_down_sync(0xffffffff, val, offset);
#     }
#     return val;
# }

# extern "C" __global__
# void gemm_tiled(const float* __restrict__ A,
#                              const float* __restrict__ B,
#                                    float* __restrict__ C,
#                              int M, int N, int K)
# {
#     // blockDim = (32, 8)示例，256线程
#     int tx = threadIdx.x;  // 0..31 (warp内线程id)
#     int ty = threadIdx.y;  // 0..7 (warp行id)
    
#     int warp_id = ty; // 每个warp负责一行的计算
#     int lane_id = tx; // warp内线程索引
    
#     int row = blockIdx.y * 8 + warp_id;   // 每个block负责8行
#     int col = blockIdx.x * 32 + lane_id;  // 每个warp负责32列
    
#     float accum = 0.0f;
    
#     for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
#         int tiled_k = t * TILE;
        
#         // 每线程加载一个元素A和B
#         float a_val = 0.0f;
#         if (row < M && (tiled_k + lane_id) < K) {
#             a_val = A[row * K + tiled_k + lane_id];
#         }
        
#         float b_val = 0.0f;
#         if ((tiled_k + warp_id) < K && col < N) {
#             b_val = B[(tiled_k + warp_id) * N + col];
#         }
        
#         // 每线程计算部分乘积
#         accum += a_val * b_val;
#     }
    
#     // warp内部累加，lane 0得到总和
#     accum = warpReduceSum(accum);
    
#     if (lane_id == 0 && row < M && col < N) {
#         C[row * N + col] = accum;
#     }
# }



# '''



kernel_code = r'''
#define TILE 16
#define WARP_SIZE 32

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__
void gemm_tiled(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int M, int N, int K)
{
    // 共享内存声明：双缓冲设计
    __shared__ __align__(16) float sA[2][TILE][TILE];
    __shared__ __align__(16) float sB[2][TILE][TILE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int warp_id = ty;
    int lane_id = tx;
    
    int row = by * 8 + warp_id;
    int col = bx * 32 + lane_id;
    
    // 使用寄存器累积结果
    float accum = 0.0f;
    
    // 预取索引
    int load_k = 0;
    int current = 0;
    int next = 1;
    
    // 预加载第一个tile
    if (load_k < K) {
        // 加载A的tile
        int load_row = row;
        int load_col = load_k + tx;
        if (load_row < M && load_col < K) {
            sA[current][ty][tx] = A[load_row * K + load_col];
        } else {
            sA[current][ty][tx] = 0.0f;
        }
        
        // 加载B的tile
        int b_load_row = load_k + ty;
        int b_load_col = col;
        if (b_load_row < K && b_load_col < N) {
            sB[current][ty][tx] = B[b_load_row * N + b_load_col];
        } else {
            sB[current][ty][tx] = 0.0f;
        }
    }
    load_k += TILE;
    
    __syncthreads();
    
    // 主循环 - 计算num_tiles次
    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // 预加载下一个tile
        if (load_k < K) {
            // 加载A的tile
            int load_row = row;
            int load_col = load_k + tx;
            if (load_row < M && load_col < K) {
                sA[next][ty][tx] = A[load_row * K + load_col];
            } else {
                sA[next][ty][tx] = 0.0f;
            }
            
            // 加载B的tile
            int b_load_row = load_k + ty;
            int b_load_col = col;
            if (b_load_row < K && b_load_col < N) {
                sB[next][ty][tx] = B[b_load_row * N + b_load_col];
            } else {
                sB[next][ty][tx] = 0.0f;
            }
        }
        load_k += TILE;
        
        // 计算当前tile
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            accum += sA[current][ty][k] * sB[current][k][tx];
        }
        
        __syncthreads();
        // 交换缓冲区
        current = 1 - current;
        next = 1 - next;
    }
    
    // warp内部归约
    accum = warpReduceSum(accum);
    
    // 写入结果
    if (lane_id == 0 && row < M && col < N) {
        C[row * N + col] = accum;
    }
}


'''

# 编译
gemm_kernel = cp.RawKernel(kernel_code, 'gemm_tiled')

# -----------------------
# 3. Launch 配置
# -----------------------
threads_per_block = (16, 16)
blocks_per_grid = ((N + 15) // 16, (M + 15) // 16)

# 预热
gemm_kernel(
    blocks_per_grid, threads_per_block,
    (d_A, d_B, d_C, np.int32(M), np.int32(N), np.int32(K))
)
cp.cuda.runtime.deviceSynchronize()

# -----------------------
# 4. 性能测试
# -----------------------
times = []
for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    gemm_kernel(
        blocks_per_grid, threads_per_block,
        (d_A, d_B, d_C, np.int32(M), np.int32(N), np.int32(K))
    )
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
    print(f"Run {i+1}: {times[-1]:.3f} ms")

print(f"Avg: {np.mean(times):.3f} ms ± {np.std(times):.3f} ms")

 

