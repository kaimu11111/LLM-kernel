import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os
import sys

# 确保文件存在
required_files = ['graphs/am.config', 'graphs/am.new_indptr', 'graphs/am.new_indices', 'graphs/am.graph.edgedump']
for file in required_files:
    if not os.path.exists(file):
        print(f"错误: 文件 '{file}' 不存在!")
        sys.exit(1)

# 1. 解析配置文件
try:
    with open('graphs/am.config', 'r') as f:
        content = f.read().strip()
        # 处理可能的空格分隔值
        if ' ' in content:
            num_nodes, nnz = map(int, content.split())
        else:
            print(f"警告: am.config 格式异常: '{content}'")
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
        indptr = np.fromfile('graphs/am.new_indptr', dtype=dtype)
        print(f"读取成功! indptr 长度: {len(indptr)}")
        
        print(f"尝试使用 {dtype} 读取 indices...")
        indices = np.fromfile('graphs/am.new_indices', dtype=dtype)
        print(f"读取成功! indices 长度: {len(indices)}")
        
        print("尝试读取 data...")
        data = np.fromfile('graphs/am.graph.edgedump', dtype=np.float32)
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
try:
    # 使用实际读取的长度调整矩阵大小
    actual_rows = len(indptr) - 1
    actual_cols = actual_rows  # 假设是方阵
    
    print(f"尝试创建稀疏矩阵 ({actual_rows} × {actual_cols})...")
    sparse_matrix = csr_matrix((data, indices, indptr), shape=(actual_rows, actual_cols))
    
    print("成功创建稀疏矩阵!")
    print(f"实际非零元素: {sparse_matrix.nnz}")
    
    # 4. 保存结果
    save_npz('sparse_matrix.npz', sparse_matrix)
    print("矩阵已保存为 sparse_matrix.npz")
    
    # 验证保存的矩阵
    try:
        loaded_matrix = csr_matrix.load('sparse_matrix.npz')
        print(f"验证成功! 加载矩阵: {loaded_matrix.shape}, nnz={loaded_matrix.nnz}")
    except Exception as e:
        print(f"验证保存的矩阵时出错: {str(e)}")
    
except Exception as e:
    print(f"创建矩阵失败: {str(e)}")
    print("尝试备用方案...")
    
    # 备用方案：直接使用读取的数据
    try:
        # 尝试使用原始配置的维度
        sparse_matrix = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
        save_npz('sparse_matrix_fallback1.npz', sparse_matrix)
        print(f"备用方案1矩阵已保存 (维度: {num_nodes} × {num_nodes})")
    except:
        # 最后备用方案：使用实际数据长度
        actual_rows = len(indptr) - 1
        actual_cols = actual_rows
        sparse_matrix = csr_matrix((data, indices, indptr), shape=(actual_rows, actual_cols))
        save_npz('sparse_matrix_fallback2.npz', sparse_matrix)
        print(f"备用方案2矩阵已保存 (维度: {actual_rows} × {actual_cols})")