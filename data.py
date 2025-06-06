import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os
import sys
dataset = "wikikg2"
# 确保文件存在
required_files = [f'graphs/{dataset}.config', f'graphs/{dataset}.new_indptr', f'graphs/{dataset}.new_indices', f'graphs/{dataset}.graph.edgedump']
for file in required_files:
    if not os.path.exists(file):
        print(f"错误: 文件 '{file}' 不存在!")
        sys.exit(1)

# 1. 解析配置文件
try:
    with open(f'graphs/{dataset}.config', 'r') as f:
        content = f.read().strip()
        # 处理可能的空格分隔值
        if ' ' in content:
            num_nodes, nnz = map(int, content.split())
        else:
            print(f"警告:{dataset}.config 格式异常: '{content}'")
            # 尝试使用默认值
            num_nodes = 881680
            nnz = 5668682
    
    print(f"矩阵维度: {num_nodes} × {num_nodes}, 非零元素: {nnz}")
except Exception as e:
    print(f"解析配置文件失败: {str(e)}")
    sys.exit(1)

# 2. 读取二进制文件
indptr = None
indices = None
data = None

# 尝试不同数据类型
for dtype in [np.int32, np.int64]:
    try:
        print(f"尝试使用 {dtype} 读取 indptr...")
        indptr = np.fromfile(f'graphs/{dataset}.new_indptr', dtype=dtype)
        print(f"读取成功! indptr 长度: {len(indptr)}")
        
        print(f"尝试使用 {dtype} 读取 indices...")
        indices = np.fromfile(f'graphs/{dataset}.new_indices', dtype=dtype)
        print(f"读取成功! indices 长度: {len(indices)}")
        
        print("尝试读取 data...")
        data = np.fromfile(f'graphs/{dataset}.graph.edgedump', dtype=np.float32)
        print(f"读取成功! data 长度: {len(data)}")
        
        # 验证数据长度
        if len(indptr) == num_nodes + 1:
            print(f"indptr 长度匹配: {len(indptr)} == {num_nodes + 1}")
        else:
            print(f"警告: indptr 长度不匹配: {len(indptr)} ≠ {num_nodes + 1}")
        
        if len(indices) == nnz:
            print(f"indices 长度匹配: {len(indices)} == {nnz}")
        else:
            print(f"警告: indices 长度不匹配: {len(indices)} ≠ {nnz}")
        
        if len(data) == nnz:
            print(f"data 长度匹配: {len(data)} == {nnz}")
        else:
            print(f"警告: data 长度不匹配: {len(data)} ≠ {nnz}")
        
        # 如果所有数据都读取成功，跳出循环
        if indptr is not None and indices is not None and data is not None:
            break
            
    except Exception as e:
        print(f"读取文件时出错 (dtype={dtype}): {str(e)}")
        continue

# 检查是否成功读取
if indptr is None or indices is None or data is None:
    print("错误: 无法读取一个或多个文件")
    sys.exit(1)

# 3. 构建稀疏矩阵
# 使用实际读取的长度调整矩阵大小
actual_rows = len(indptr) - 1
actual_cols = actual_rows  # 假设是方阵
    
print(f"尝试创建稀疏矩阵 ({actual_rows} × {actual_cols})...")
sparse_matrix = csr_matrix((data, indices, indptr), shape=(actual_rows, actual_cols))
    
print("成功创建稀疏矩阵!")
print(f"实际非零元素: {sparse_matrix.nnz}")
    
# 4. 保存结果
save_npz(f'sparse_matrix_{dataset}.npz', sparse_matrix)
print("矩阵已保存为 sparse_matrix.npz")
