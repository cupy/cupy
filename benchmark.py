import numpy as np
import cupy as cp
from cupy import xpu # from cupy import cuda as xpu
import time

# 配置参数
dtype = np.float32  # 使用float32以获得最佳GPU性能
N = 200000000       # 向量长度200M
M = 4000            # 矩阵尺寸4K x 4K
repeating = 10
use_async = False
mode = "AsyncMode异步模式" if use_async else "SyncMode同步模式"


def warmup(gpu_func, gpu_data):
    """
    预热GPU：通过运行一次操作来初始化GPU上下文，避免首次运行的开销。
    """
    if gpu_data is not None:
        _ = gpu_func(*gpu_data)
        xpu.Stream.null.synchronize()  # 确保操作完成

def benchmark_vector_op(np_func, cp_func, np_vector1, np_vector2, cp_vector1, cp_vector2, 
                        unary=True, repeating=10, use_async=False):
    """
    基准测试向量操作在NumPy（CPU）和CuPy（GPU）上的性能，支持单目和双目运算。
    
    Args:
        np_func: NumPy函数（如np.cos或np.add）
        cp_func: CuPy函数（如cp.cos或cp.add）
        np_vector1: NumPy向量1（CPU数据）
        np_vector2: NumPy向量2（CPU数据，用于双目运算）
        cp_vector1: CuPy向量1（GPU数据）
        cp_vector2: CuPy向量2（GPU数据，用于双目运算）
        unary: 是否为单目运算（True为单目，False为双目）
        repeating: 重复测试次数（默认10）
        use_async: 是否使用异步模式（仅影响CuPy）
    
    Returns:
        tuple: (numpy_avg_time, cupy_avg_time, speedup_ratio)
    """
    # 预热GPU
    warmup(cp_func, (cp_vector1,) if unary else (cp_vector1, cp_vector2))
    
    # 根据运算类型选择参数
    np_args = (np_vector1,) if unary else (np_vector1, np_vector2)
    cp_args = (cp_vector1,) if unary else (cp_vector1, cp_vector2)
    
    # 测试NumPy（CPU，始终同步执行）
    np_times = []
    for _ in range(repeating):
        start = time.perf_counter()
        cpu_result = np_func(*np_args)
        end = time.perf_counter()
        np_times.append(end - start)
    np_avg = np.mean(np_times)
    
    # 测试CuPy（GPU，根据use_async选择模式）
    cp_times = []
    if not use_async:
        # 同步模式：使用默认流并显式同步
        for _ in range(repeating):
            start = time.perf_counter()
            xpu_result = cp_func(*cp_args)
            xpu.Stream.null.synchronize()  # 阻塞直到GPU完成
            end = time.perf_counter()
            cp_times.append(end - start)
    else:
        # 异步模式：使用非阻塞流
        stream = xpu.Stream()  # 创建新流
        for _ in range(repeating):
            start = time.perf_counter()
            with stream:
                xpu_result = cp_func(*cp_args)  # 在流中提交操作
            end = time.perf_counter()
            cp_times.append(end - start)
        stream.synchronize()  # 等待该流中的操作完成
    cp_avg = np.mean(cp_times)
    
    # 计算加速比
    print("cpu result: ", cpu_result)
    print("xpu result: ", xpu_result)
    speedup_ratio = np_avg / cp_avg if cp_avg > 0 else 0
    print(f"NumPy 平均时间: {np_avg:.6f} 秒")
    print(f"CuPy (同步) 平均时间: {cp_avg:.6f} 秒")
    print(f"加速比: {speedup_ratio:.2f}x")
    return np_avg, cp_avg, speedup_ratio

def benchmark_mat_op(np_func, cp_func, np_mat_a, np_mat_b, cp_mat_a, cp_mat_b, 
                     unary=True, repeating=10, use_async=False):
    """
    基准测试矩阵操作在NumPy（CPU）和CuPy（GPU）上的性能，支持单目运算和矩阵乘法。
    
    Args:
        np_func: NumPy函数（如np.cos或np.dot）
        cp_func: CuPy函数（如cp.cos或cp.dot）
        np_mat_a: NumPy矩阵A（CPU数据）
        np_mat_b: NumPy矩阵B（CPU数据，用于矩阵乘法）
        cp_mat_a: CuPy矩阵A（GPU数据）
        cp_mat_b: CuPy矩阵B（GPU数据，用于矩阵乘法）
        unary: 是否为单目运算（True为单目，False为矩阵乘法）
        repeating: 重复测试次数（默认10）
        use_async: 是否使用异步模式（仅影响CuPy）
    
    Returns:
        tuple: (numpy_avg_time, cupy_avg_time, speedup_ratio)
    """
    # 预热GPU
    warmup(cp_func, (cp_mat_a,) if unary else (cp_mat_a, cp_mat_b))
    
    # 根据运算类型选择参数
    np_args = (np_mat_a,) if unary else (np_mat_a, np_mat_b)
    cp_args = (cp_mat_a,) if unary else (cp_mat_a, cp_mat_b)
    
    # 测试NumPy（CPU）
    np_times = []
    for _ in range(repeating):
        start = time.perf_counter()
        cpu_result = np_func(*np_args)
        end = time.perf_counter()
        np_times.append(end - start)
    np_avg = np.mean(np_times)
    
    # 测试CuPy（GPU）
    cp_times = []
    if not use_async:
        for _ in range(repeating):
            start = time.perf_counter()
            xpu_result = cp_func(*cp_args)
            xpu.Stream.null.synchronize()
            end = time.perf_counter()
            cp_times.append(end - start)
    else:
        stream = xpu.Stream()
        for _ in range(repeating):
            start = time.perf_counter()
            with stream:
                xpu_result = cp_func(*cp_args)
            stream.synchronize()
            end = time.perf_counter()
            cp_times.append(end - start)
    cp_avg = np.mean(cp_times)
    
    # 计算加速比
    speedup_ratio = np_avg / cp_avg if cp_avg > 0 else 0
    
    print(f"CPU result: ", cpu_result)
    print(f"XPU result: ", xpu_result)
    print(f"NumPy 平均时间: {np_avg:.6f} 秒")
    print(f"CuPy (同步) 平均时间: {cp_avg:.6f} 秒")
    print(f"加速比: {speedup_ratio:.2f}x")
    return np_avg, cp_avg, speedup_ratio

if __name__ == "__main__":
    
    # 定义测试函数
    np_cos, cp_cos = np.cos, cp.cos
    np_add, cp_add = np.add, cp.add
    np_dot, cp_dot = np.matmul, cp.matmul
    np_sum, cp_sum = np.sum, cp.sum

    print("正在生成测试数据...（这可能需要一些时间）")
    
    # 生成向量数据（200M元素）
    try:
        np_vec1 = np.random.rand(N).astype(dtype)
        np_vec2 = np.random.rand(N).astype(dtype)
        np_vec_ret = np.random.rand(N).astype(dtype)
        cp_vec1 = cp.asarray(np_vec1)  # 将NumPy数组转换为CuPy数组
        cp_vec2 = cp.asarray(np_vec2)
        cp_vec_ret = cp.asarray(np_vec_ret)
        print(f"向量数据生成完成：NumPy数组形状 {np_vec1.shape}, CuPy数组形状 {cp_vec1.shape}")
    except Exception as e:
        print(f"生成向量数据时出错：{e}。请检查可用内存/显存。")
        exit(1)
    
    # 生成矩阵数据（4K x 4K）
    try:
        np_mat_a = np.random.rand(M, M).astype(dtype)
        np_mat_b = np.random.rand(M, M).astype(dtype)
        cp_mat_a = cp.asarray(np_mat_a)
        cp_mat_b = cp.asarray(np_mat_b)
        print(f"矩阵数据生成完成：NumPy数组形状 {np_mat_a.shape}, CuPy数组形状 {cp_mat_a.shape}")
    except Exception as e:
        print(f"生成矩阵数据时出错：{e}。请检查可用内存/显存。")
        exit(1)

    # function unit test
    print("\n" + "="*60)
    cp_vec_ret += cp_vec1
    print("test inplace add op, with result", cp_vec_ret)
    print("="*60)

    print("\n" + "="*60)
    cp_vec_ret += cp.float32(2.0)
    print("test tensor + sclar op, with result", cp_vec_ret)
    print("="*60)

    print("\n" + "="*60)
    _ = cp.add(cp_mat_a, cp_mat_b)
    print("test dot op, with result", cp_vec_ret)
    print("="*60)

    print("\n" + "="*60)
    cp_vec_ret = cp.dot(cp_vec1, cp_vec2)
    print("test dot op, with result", cp_vec_ret)
    print("="*60)
        
    print("\n" + "="*60)
    print(f"基准测试向量COS操作（单目运算，{mode}）")
    print("="*60)
    np_time, cp_time, speedup = benchmark_vector_op(np_cos, cp_cos, np_vec1, None, cp_vec1, None, 
                                                    unary=True, repeating=repeating, use_async=use_async)
    
    print("\n" + "="*60)
    print(f"基准测试向量Sum操作（Reduction运算， {mode}）")
    print("="*60)
    np_time, cp_time, speedup = benchmark_vector_op(np_sum, cp_sum, np_vec1, None, cp_vec1, None, 
                                                    unary=True, repeating=repeating, use_async=use_async)

    print("\n" + "="*60)
    print(f"基准测试向量加法操作（双目运算，{mode}）")
    print("="*60)
    np_time, cp_time, speedup = benchmark_vector_op(np_add, cp_add, np_vec1, np_vec2, cp_vec1, cp_vec2, 
                                                    unary=False, repeating=repeating, use_async=use_async)
    
    print("\n" + "="*60)
    print(f"基准测试矩阵乘法操作（双目运算，{mode}）")
    print("="*60)
    np_time, cp_time, speedup = benchmark_mat_op(np_dot, cp_dot, np_mat_a, np_mat_b, cp_mat_a, cp_mat_b, 
                                                unary=False, repeating=repeating, use_async=use_async)
    
    print("\n" + "="*60)
    print(f"基准测试Vector乘法操作（双目运算，{mode}）")
    print("="*60)
    np_time, cp_time, speedup = benchmark_mat_op(np.multiply, cp.multiply, np_vec1, np_vec2, cp_vec1, cp_vec2, 
                                                unary=False, repeating=repeating, use_async=use_async)

    # print("\n" + "="*60) # cos, add is not supported for dim = 2 matrix for ASCEND
    # print("基准测试矩阵COS操作（单目运算，同步模式）")
    # print("="*60)
    # np_time, cp_time, speedup = benchmark_mat_op(np_cos, cp_cos, np_mat_a, None, cp_mat_a, None, 
    #                                              unary=True, repeating=repeating, use_async=use_async)