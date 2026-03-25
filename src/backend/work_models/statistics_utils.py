import json
import os
import glob
import base64
import io
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def calculate_transformation(metrics: Dict) -> float:
    """
    根据workflow.py中的计算规则计算transformation
    
    Args:
        metrics: JSON文件中的metrics字段
        
    Returns:
        计算出的transformation值
    """
    w_time = 0.4
    w_gpu = 0.3
    w_mem = 0.3
    
    r_time = metrics.get('performance', {}).get('ratios', {}).get('response_time', 0)
    r_gpu = metrics.get('performance', {}).get('ratios', {}).get('gpu_utilization', 0)
    r_mem = metrics.get('performance', {}).get('ratios', {}).get('memory_usage', 0)
    
    transformation = 0.0
    if r_time != 0:
        transformation += w_time / r_time
    if r_gpu != 0:
        transformation += w_gpu / r_gpu
    if r_mem != 0:
        transformation += w_mem / r_mem
    
    return round(transformation, 4)

def load_rag_results(results_dir: str) -> Dict[str, List[Dict]]:
    """
    加载所有RAG结果文件，并确保transformation字段存在
    
    Args:
        results_dir: rag_results文件夹路径
        
    Returns:
        按模型和领域分组的结果数据
    """
    results = {}
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_name = data.get('model_name', '')
            domain = data.get('domain', '')
            rule = data.get('rule', '')
            
            if model_name and domain:
                # 检查并计算transformation
                metrics = data.get('metrics', {})
                if 'transformation' not in metrics and 'performance' in metrics:
                    metrics['transformation'] = calculate_transformation(metrics)
                    data['metrics'] = metrics
                
                key = f"{model_name}_{domain}_{rule}"
                if key not in results:
                    results[key] = []
                results[key].append(data)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def calculate_domain_stats(results: Dict[str, List[Dict]], domain: str, model_name: str) -> Dict:
    """
    计算特定模型在特定领域的统计数据
    
    Args:
        results: 所有结果数据
        domain: 领域名称
        model_name: 模型名称
        
    Returns:
        该领域该模型的统计结果
    """
    stats = {
        'basic_accuracy': 0.0,
        'rag_accuracy': 0.0,
        'improvement': 0.0,
        'transformation': 0.0,
        'dataset_size': 0
    }
    
    # 查找该模型该领域的所有结果文件
    domain_results = []
    for key, file_results in results.items():
        if key.startswith(f"{model_name}_{domain}_"):
            domain_results.extend(file_results)
    
    if not domain_results:
        return stats
    
    # 计算平均值
    total_basic = 0.0
    total_rag = 0.0
    total_improvement = 0.0
    total_transformation = 0.0
    total_dataset_size = 0
    
    for result in domain_results:
        metrics = result.get('metrics', {})
        parameters = result.get('parameters', {})
        total_basic += metrics.get('basic_accuracy', 0.0)
        total_rag += metrics.get('rag_accuracy', 0.0)
        total_improvement += metrics.get('improvement', 0.0)
        total_transformation += metrics.get('transformation', 0.0)
        total_dataset_size += parameters.get('dataset_size', 0)
    
    file_count = len(domain_results)
    stats.update({
        'basic_accuracy': round(total_basic / file_count, 4),
        'rag_accuracy': round(total_rag / file_count, 4),
        'improvement': round(total_improvement / file_count, 4),
        'transformation': round(total_transformation / file_count, 4),
        'dataset_size': total_dataset_size  # 改为总的dataset_size而不是文件数量
    })
    
    return stats

def get_model_statistics(results: Dict[str, List[Dict]], selected_model_id: str) -> Dict:
    """
    获取指定模型的统计结果
    
    Args:
        results: 所有结果数据
        selected_model_id: 选择的模型ID
        
    Returns:
        指定模型在所有领域的统计结果
    """
    # 根据workflow.py的模型映射 (用于数据过滤)
    model_map = {
        "1": "qwen",      # Qwen/Qwen-1_8B
        "2": "gpt2",      # gpt2-medium
        "3": "gptneo",    # EleutherAI/gpt-neo-125M
        "5": "opt",       # facebook/opt-1.3b
        "api": "api"      # deepseek API (用于文件匹配)
    }

    # 显示名称映射 (用于UI显示)
    display_name_map = {
        "1": "Qwen-1.8B",      # Qwen/Qwen-1_8B
        "2": "GPT-2 Medium",    # gpt2-medium
        "3": "GPT-Neo-125M",    # EleutherAI/gpt-neo-125M
        "5": "OPT-1.3B",        # facebook/opt-1.3b
        "api": "DeepSeek-Chat"  # deepseek API
    }
    
    domains = [
        'culture', 'geography', 'health', 'history', 'mathematics',
        'nature', 'people', 'society', 'technology'
    ]
    
    if selected_model_id not in display_name_map:
        return {"error": "Invalid model ID"}

    display_name = display_name_map[selected_model_id]
    file_match_name = model_map[selected_model_id]  # 用于文件匹配的名称

    statistics = {
        "model_id": selected_model_id,
        "model_name": display_name,  # 用于显示的名称
        "domains": {}
    }

    for domain in domains:
        domain_stats = calculate_domain_stats(results, domain, file_match_name)  # 使用文件匹配名称
        statistics["domains"][domain] = domain_stats
    
    return statistics

def print_model_statistics(results: Dict[str, List[Dict]], selected_model_id: str):
    """
    打印指定模型的统计结果（用于调试）
    
    Args:
        results: 所有结果数据
        selected_model_id: 选择的模型ID
    """
    stats = get_model_statistics(results, selected_model_id)
    
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return
    
    print("=" * 80)
    print(f"RAG RESULTS STATISTICS - MODEL {stats['model_id']}")
    print("=" * 80)
    
    print(f"📊 MODEL: {stats['model_name']} (ID: {stats['model_id']})")
    print("-" * 80)
    
    total_basic = 0.0
    total_rag = 0.0
    total_improvement = 0.0
    total_transformation = 0.0
    total_dataset_size = 0
    
    for domain, domain_stats in stats['domains'].items():
        total_basic += domain_stats['basic_accuracy']
        total_rag += domain_stats['rag_accuracy']
        total_improvement += domain_stats['improvement']
        total_transformation += domain_stats['transformation']
        total_dataset_size += domain_stats['dataset_size']
        
        print(f"  {domain.upper():<15}: "
              f"Basic: {domain_stats['basic_accuracy']:.4f} | "
              f"RAG: {domain_stats['rag_accuracy']:.4f} | "
              f"Improve: {domain_stats['improvement']:.4f} | "
              f"Transform: {domain_stats['transformation']:.4f} | "
              f"Dataset: {domain_stats['dataset_size']}")
    

def main():
    # statistics_utils.py现在在work_models文件夹下
    # 需要向上导航到src/backend/，然后到experiments/results/rag_results/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rag_results_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'experiments', 'results', 'rag_results'))
    
    # 加载所有结果
    results = load_rag_results(rag_results_dir)
    
    print(f"Loaded {len(results)} result files")
    
    # 演示：打印模型1(qwen)的统计结果
    print("\n=== Statistics for Model 1 (Qwen) ===")
    print_model_statistics(results, "1")

def plot_radar_chart(results: Dict[str, List[Dict]], selected_model_id: str, save_path: str = None) -> str:
    """
    生成指定模型的雷达图，比较basic和rag在各领域的性能
    
    Args:
        results: 所有结果数据
        selected_model_id: 选择的模型ID
        save_path: 保存图片的路径，如果为None则返回base64编码
        
    Returns:
        如果save_path为None，返回base64编码的图片数据；否则返回保存的文件路径
    """
    try:
        # 设置matplotlib使用非GUI后端，避免线程问题
        import matplotlib
        matplotlib.use('Agg')  # 使用非GUI后端
        
        # 获取模型统计数据
        stats = get_model_statistics(results, selected_model_id)
        
        if "error" in stats:
            print(f"Error getting statistics: {stats['error']}")
            return None
        
        # 领域列表
        domains = [
            'culture', 'geography', 'health', 'history', 'mathematics',
            'nature', 'people', 'society', 'technology'
        ]
        
        # 准备数据
        basic_scores = []
        rag_scores = []
        domain_labels = []
        
        for domain in domains:
            if domain in stats['domains']:
                domain_stats = stats['domains'][domain]
                basic_scores.append(domain_stats['basic_accuracy'])
                rag_scores.append(domain_stats['rag_accuracy'])
                # 将领域名称首字母大写作为标签
                domain_labels.append(domain.capitalize())
            else:
                # 如果某个领域没有数据，设为0
                basic_scores.append(0.0)
                rag_scores.append(0.0)
                domain_labels.append(domain.capitalize())
        
        # 设置雷达图参数
        num_vars = len(domain_labels)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # 使雷达图闭合
        basic_scores_plot = basic_scores + [basic_scores[0]]
        rag_scores_plot = rag_scores + [rag_scores[0]]
        angles_plot = angles + [angles[0]]
        domain_labels_plot = domain_labels + [domain_labels[0]]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 绘制basic accuracy雷达图
        line1 = ax.plot(angles_plot, basic_scores_plot, 'o-', linewidth=3, label='Basic Accuracy', color='blue', alpha=0.8, markersize=8)
        ax.fill(angles_plot, basic_scores_plot, alpha=0.2, color='blue')
        
        # 绘制rag accuracy雷达图
        line2 = ax.plot(angles_plot, rag_scores_plot, 'o-', linewidth=3, label='RAG Accuracy', color='red', alpha=0.8, markersize=8)
        ax.fill(angles_plot, rag_scores_plot, alpha=0.2, color='red')
        
        # 为每个数据点添加数值标签
        for i in range(num_vars):
            # Basic accuracy标签
            basic_value = basic_scores[i]
            basic_angle = angles[i]
            ax.annotate(f'{basic_value:.3f}', 
                       xy=(basic_angle, basic_value), 
                       xytext=(basic_angle, basic_value + 0.05),
                       fontsize=8, fontweight='bold', color='blue',
                       ha='center', va='bottom')
            
            # RAG accuracy标签
            rag_value = rag_scores[i]
            rag_angle = angles[i]
            ax.annotate(f'{rag_value:.3f}', 
                       xy=(rag_angle, rag_value), 
                       xytext=(rag_angle, rag_value + 0.05),
                       fontsize=8, fontweight='bold', color='red',
                       ha='center', va='bottom')
        
        # 添加标签
        ax.set_xticks(angles)
        ax.set_xticklabels(domain_labels, fontsize=11, fontweight='bold')
        
        # 设置y轴标签和范围
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold', labelpad=20)
        
        # 添加标题 - 使用stats中的model_name（已经在get_model_statistics中设置了正确的显示名称）
        ax.set_title(f'Performance Comparison: {stats["model_name"]}\nBasic vs RAG Accuracy by Domain',
                    size=16, fontweight='bold', pad=30)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=11, framealpha=0.9)
        
        # 添加网格线
        ax.grid(True, alpha=0.3, linewidth=1)
        
        # 添加同心圆参考线
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=0, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加性能摘要文本
        avg_basic = sum(basic_scores) / len(basic_scores)
        avg_rag = sum(rag_scores) / len(rag_scores)
        improvement = avg_rag - avg_basic
        
        summary_text = f'Avg Basic: {avg_basic:.4f} | Avg RAG: {avg_rag:.4f} | Improvement: {improvement:.4f}'
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        if save_path:
            # 保存图片到文件
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            return save_path
        else:
            # 返回base64编码的图片数据
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return img_base64
            
    except Exception as e:
        print(f"Error generating radar chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_model_radar_chart(selected_model_id: str, output_format: str = 'base64') -> str:
    """
    生成指定模型的雷达图并返回图片数据
    
    Args:
        selected_model_id: 模型ID ("1", "2", "3", "5")
        output_format: 输出格式 ('base64' 或 'file')
        
    Returns:
        base64编码的图片数据或文件路径
    """
    # 获取statistics_utils.py所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # rag_results文件夹路径
    rag_results_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'experiments', 'results', 'rag_results'))
    
    try:
        # 加载统计数据
        results = load_rag_results(rag_results_dir)
        
        if output_format == 'file':
            # 生成文件路径
            import time
            timestamp = str(int(time.time()))
            output_file = os.path.join(current_dir, '..', 'static', 'images', f'radar_chart_model_{selected_model_id}_{timestamp}.png')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            return plot_radar_chart(results, selected_model_id, output_file)
        else:
            # 返回base64编码
            return plot_radar_chart(results, selected_model_id)
            
    except Exception as e:
        print(f"Error in generate_model_radar_chart: {e}")
        return None

if __name__ == "__main__":
    # 测试雷达图生成
    import time
    
    # 生成模型1的雷达图
    print("Generating radar chart for Model 1...")
    result = generate_model_radar_chart("1", "base64")
    
    if result:
        print(f"Radar chart generated successfully! Base64 length: {len(result)} characters")
    else:
        print("Failed to generate radar chart")