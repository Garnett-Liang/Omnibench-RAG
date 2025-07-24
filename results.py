import json
import os

# 定义各效率指标的权重（可根据实际需求调整）
WEIGHTS = {
    'response_time': 0.4,  # 响应时间权重最高，影响用户体验
    'memory_usage': 0.3,   # 内存使用权重次之
    'gpu_utilization': 0.3  # GPU利用率权重与内存相当
}

def calculate_transformation(ratios):
    """
    根据效率指标变化率计算综合转换评分（transformation）
    变化率 < 1 表示效率提升（正向转换），>1 表示效率下降（负向转换）
    评分公式：Σ(1/变化率 * 权重) → 数值越高，综合效率提升越显著
    """
    if not ratios:
        return 0.0
    transformation = 0.0
    for metric, ratio in ratios.items():
        if ratio <= 0:  # 避免除以0或负变化率（理论上变化率应为正数）
            continue
        # 变化率越小（效率提升越大），该项得分越高
        transformation += (1 / ratio) * WEIGHTS.get(metric, 0)
    return round(transformation, 4)

# 初始化结果存储
results = {}
directory = 'rag_results'

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            domain = data['domain']
            params = data['parameters']
            metrics = data['metrics']
            perf = metrics['performance']
            
            # 提取基础数据
            dataset_size = params['dataset_size']
            basic_acc = metrics['basic_accuracy']
            rag_acc = metrics['rag_accuracy']
            
            # 提取效率指标变化率
            ratios = perf['ratios']  # 包含 response_time/memory_usage/gpu_utilization 的变化率
            
            # 初始化领域数据
            if domain not in results:
                results[domain] = {
                    'total_size': 0,
                    'total_basic_correct': 0,
                    'total_rag_correct': 0,
                    # 效率变化率按样本量加权累加
                    'sum_time_ratio': 0,
                    'sum_memory_ratio': 0,
                    'sum_gpu_ratio': 0
                }
            
            # 累加数据（按样本量加权）
            results[domain]['total_size'] += dataset_size
            results[domain]['total_basic_correct'] += basic_acc * dataset_size
            results[domain]['total_rag_correct'] += rag_acc * dataset_size
            results[domain]['sum_time_ratio'] += ratios['response_time'] * dataset_size
            results[domain]['sum_memory_ratio'] += ratios['memory_usage'] * dataset_size
            results[domain]['sum_gpu_ratio'] += ratios['gpu_utilization'] * dataset_size

# 计算整合结果
print(f"各领域综合评估结果（含效率转换评分）\n")
print(f"{'领域':<12} | 问题总数 | 基础准确率 | RAG准确率 | 准确率变化 | 综合效率转换评分")
print("-" * 100)

for domain, info in results.items():
    total = info['total_size']
    if total == 0:
        continue
    
    # 计算准确率
    basic_acc = (info['total_basic_correct'] / total) * 100
    rag_acc = (info['total_rag_correct'] / total) * 100
    acc_change = rag_acc - basic_acc
    
    # 计算加权平均变化率
    avg_time_ratio = info['sum_time_ratio'] / total
    avg_memory_ratio = info['sum_memory_ratio'] / total
    avg_gpu_ratio = info['sum_gpu_ratio'] / total
    
    # 计算综合效率转换评分
    avg_ratios = {
        'response_time': avg_time_ratio,
        'memory_usage': avg_memory_ratio,
        'gpu_utilization': avg_gpu_ratio
    }
    transformation = calculate_transformation(avg_ratios)
    
    # 输出结果
    print(
        f"{domain:<12} | {total:<8} | {basic_acc:.1f}%      | {rag_acc:.1f}%      | "
        f"{acc_change:+.1f}%     | {transformation:.4f}"
    )

input("\n按任意键退出...")