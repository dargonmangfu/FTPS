import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import processing_times, machine_requirements, precedence_constraints
import re
import os
import time
import datetime

# 设置图片保存目录
IMAGE_DIR = r"E:\调度问题\图片"
# 确保图片保存目录存在
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"创建图片保存目录: {IMAGE_DIR}")

# 生成时间戳，用于文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 添加图表计数器字典
chart_counters = {
    "gantt": 0,
    "timeline": 0,
    "utilization": 0,
    "evolution": 0, # 添加进化过程图表计数器
    "pareto": 0  # 添加帕累托前沿图表计数器
}

def clean_filename(title):
    """将标题转换为合法的文件名"""
    # 移除非法字符
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    # 替换空格为下划线
    cleaned = cleaned.replace(" ", "_")
    return cleaned

def plot_schedule(schedule, makespan, machines_usage, n_operations, n_machines):
    """
    绘制人员调度甘特图
    
    参数:
        schedule: 操作开始时间数组
        makespan: 总完成时间
        machines_usage: 机器使用记录 (机器ID, 开始时间, 结束时间, 操作ID)
        n_operations: 操作数量
        n_machines: 机器数量
    """
    plt.figure(figsize=(16, 10))
    
    # 使用不同颜色区分不同操作
    colors = plt.cm.tab20(np.linspace(0, 1, n_operations))
    
    # 按机器ID整理操作
    machine_tasks = [[] for _ in range(n_machines)]
    for machine_id, start, end, op_id in machines_usage:
        machine_tasks[machine_id].append((start, end, op_id))
    
    # 按照开始时间排序每台机器的任务
    for m in range(n_machines):
        machine_tasks[m].sort(key=lambda x: x[0])
    
    # 绘制甘特图
    for machine_id in range(n_machines):
        y_pos = machine_id
        for start, end, op_id in machine_tasks[machine_id]:
            duration = end - start
            # 绘制操作条形
            rect = plt.barh(y_pos, duration, left=start, 
                           color=colors[op_id % len(colors)], 
                           edgecolor='black', alpha=0.8, height=0.6)
            
            # 添加操作ID标签
            if duration > makespan * 0.02:  # 只在足够宽的条形上显示文字
                plt.text(start + duration/2, y_pos, f'Op{op_id}', 
                        ha='center', va='center', color='white', 
                        fontweight='bold', fontsize=9)
    
    # 设置图表属性
    title = "Operation Schedule Gantt Chart"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Machine/Worker", fontsize=14)
    plt.yticks(range(n_machines), [f"Machine {i+1}" for i in range(n_machines)])
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.xlim(0, makespan * 1.05)  # 稍微留些空间
    plt.ylim(-0.5, n_machines - 0.5)
    
    # 添加图例
    legend_elements = []
    shown_ops = set()
    for machine_id in range(n_machines):
        for start, end, op_id in machine_tasks[machine_id]:
            if op_id not in shown_ops and len(shown_ops) < 10:  # 限制图例数量
                legend_elements.append(plt.Rectangle((0,0),1,1, 
                                     color=colors[op_id % len(colors)], 
                                     label=f'Operation {op_id}'))
                shown_ops.add(op_id)
    
    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 更新计数器并保存图片
    chart_counters["gantt"] += 1
    filename = os.path.join(IMAGE_DIR, f"{clean_filename(title)}_{timestamp}_{chart_counters['gantt']}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"甘特图已保存为: {filename}")
    plt.show()

def plot_operation_timeline(schedule, n_operations):
    """
    绘制操作时间线图
    
    参数:
        schedule: 操作开始时间数组
        n_operations: 操作数量
    """
    plt.figure(figsize=(14, 8))
    
    # 计算操作完成时间
    completion_times = []
    for op in range(n_operations):
        start_time = schedule[op]
        duration = processing_times[(1, op)]
        completion_times.append(start_time + duration)
    
    # 按开始时间排序
    sorted_ops = sorted(range(n_operations), key=lambda x: schedule[x])
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_operations))
    
    for i, op in enumerate(sorted_ops):
        start = schedule[op]
        duration = processing_times[(1, op)]
        
        # 绘制操作条
        plt.barh(i, duration, left=start, color=colors[op], alpha=0.7, edgecolor='black')
        
        # 添加操作标签
        plt.text(start + duration/2, i, f'Op{op}', 
                ha='center', va='center', fontweight='bold')
        
        # 添加前驱约束线
        if op in precedence_constraints:
            for pred in precedence_constraints[op]:
                pred_completion = schedule[pred] + processing_times[(1, pred)]
                pred_index = sorted_ops.index(pred)
                
                # 绘制依赖关系箭头
                plt.annotate('', xy=(start, i), xytext=(pred_completion, pred_index),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.6, lw=1))
    
    # 设置标题变量
    title = "Operations Timeline with Precedence Constraints"
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Operations (sorted by start time)", fontsize=12)
    plt.yticks(range(len(sorted_ops)), [f'Op{op}' for op in sorted_ops])
    plt.grid(axis='x', alpha=0.3)
    
    # 更新计数器并保存图片
    chart_counters["timeline"] += 1
    filename = os.path.join(IMAGE_DIR, f"{clean_filename(title)}_{timestamp}_{chart_counters['timeline']}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"时间线图已保存为: {filename}")
    plt.show()

def analyze_machine_utilization(machines_usage, makespan, n_machines):
    """
    分析机器利用率
    
    参数:
        machines_usage: 机器使用记录
        makespan: 总完成时间
        n_machines: 机器数量
    """
    machine_busy_time = [0] * n_machines
    
    # 计算每台机器的忙碌时间
    for machine_id, start, end, op_id in machines_usage:
        machine_busy_time[machine_id] += (end - start)
    
    # 计算利用率
    utilization_rates = [busy_time / makespan for busy_time in machine_busy_time]
    
    # 绘制利用率图
    plt.figure(figsize=(12, 6))
    
    # 设置标题变量
    title = "Machine Utilization Analysis"
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(n_machines), utilization_rates, color='steelblue', alpha=0.7)
    plt.title("Machine Utilization Rates", fontsize=14)
    plt.xlabel("Machine ID", fontsize=12)
    plt.ylabel("Utilization Rate", fontsize=12)
    plt.xticks(range(n_machines), [f'M{i+1}' for i in range(n_machines)])
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    plt.pie(machine_busy_time, labels=[f'M{i+1}' for i in range(n_machines)], 
           autopct='%1.1f%%', startangle=90)
    plt.title("Machine Workload Distribution", fontsize=14)
    
    # 更新计数器并保存图片
    chart_counters["utilization"] += 1
    filename = os.path.join(IMAGE_DIR, f"{clean_filename(title)}_{timestamp}_{chart_counters['utilization']}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"利用率分析图已保存为: {filename}")
    plt.show()
    
    # 打印统计信息
    print(f"\n机器利用率统计:")
    print(f"平均利用率: {np.mean(utilization_rates):.3f}")
    print(f"最高利用率: {np.max(utilization_rates):.3f} (Machine {np.argmax(utilization_rates)+1})")
    print(f"最低利用率: {np.min(utilization_rates):.3f} (Machine {np.argmin(utilization_rates)+1})")
    print(f"利用率标准差: {np.std(utilization_rates):.3f}")

def plot_evolution_process(best_makespan, avg_makespan, title="Algorithm Evolution Process"):
    """
    绘制算法进化过程图表
    
    参数:
        best_makespan: 每代的最佳makespan值列表
        avg_makespan: 每代的平均makespan值列表
        title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    generations = range(1, len(best_makespan) + 1)
    
    plt.plot(generations, best_makespan, 'r-', label='Best Makespan', linewidth=2)
    plt.plot(generations, avg_makespan, 'b--', label='Average Makespan', linewidth=1.5)
    
    # 删除当前makespan作为水平参考线的代码
    
    # 添加改进率标注
    if len(best_makespan) > 1:
        improvement = (best_makespan[0] - best_makespan[-1]) / best_makespan[0] * 100
        plt.annotate(f'Total Improvement: {improvement:.2f}%', 
                     xy=(len(best_makespan), best_makespan[-1]),
                     xytext=(len(best_makespan) * 0.7, 
                             best_makespan[0] - (best_makespan[0] - best_makespan[-1]) * 0.3),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                     fontsize=10, color='green')
    
    plt.title(title, fontsize=15)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 更新计数器并保存图片
    chart_counters["evolution"] += 1
    filename = os.path.join(IMAGE_DIR, f"{clean_filename(title)}_{timestamp}_{chart_counters['evolution']}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"进化过程图已保存为: {filename}")
    plt.show()

def plot_pareto_front(objectives, best_solutions=None, highlight_idx=None, title="Pareto Front Visualization"):
    """
    绘制帕累托前沿图
    
    参数:
        objectives: 二维数组，种群的目标函数值 [种群大小 x 目标数]
        best_solutions: 可选，一组特定的最优解目标值
        highlight_idx: 可选，要在图中突出显示的解的索引
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    # 确保objectives是numpy数组
    objectives = np.array(objectives)
    
    # 使用util.py中的NDsort找出非支配解
    from util import NDsort
    fronts, _ = NDsort(objectives, len(objectives), objectives.shape[1])
    pareto_indices = np.where(fronts == 1)[0]
    pareto_solutions = objectives[pareto_indices]
    
    # 绘制所有解 
    plt.scatter(objectives[:, 0], objectives[:, 1], c='lightgray', alpha=0.5, label='All Solutions')
    
    # 绘制帕累托前沿 
    plt.scatter(pareto_solutions[:, 0], pareto_solutions[:, 1], c='blue', s=80, label='Pareto Optimal Solutions')
    
    # 连接帕累托前沿上的点（按makespan排序）
    sorted_pareto = pareto_solutions[pareto_solutions[:, 0].argsort()]
    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'b--', alpha=0.3)
    
    # 如果提供了特定的最优解集合，也绘制它们
    if best_solutions is not None:
        best_solutions = np.array(best_solutions)
        plt.scatter(best_solutions[:, 0], best_solutions[:, 1], c='green', s=100, marker='*', label='Selected Best Solutions')
    
    # 突出显示特定解
    if highlight_idx is not None and highlight_idx < len(objectives):
        plt.scatter(objectives[highlight_idx, 0], objectives[highlight_idx, 1], 
                  c='red', s=150, marker='X', label='Current Selected Solution')
        
        # 添加标注
        plt.annotate(f'Makespan: {objectives[highlight_idx, 0]:.2f}\nImbalance: {objectives[highlight_idx, 1]:.4f}',
                   xy=(objectives[highlight_idx, 0], objectives[highlight_idx, 1]),
                   xytext=(objectives[highlight_idx, 0] + (max(objectives[:, 0]) - min(objectives[:, 0]))*0.05, 
                           objectives[highlight_idx, 1]),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel("Objective 1: Completion Time (Makespan)", fontsize=12)
    plt.ylabel("Objective 2: Resource Utilization Imbalance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # 更新计数器并保存图片
    chart_counters["pareto"] = chart_counters.get("pareto", 0) + 1
    filename = os.path.join(IMAGE_DIR, f"{clean_filename(title)}_{timestamp}_{chart_counters['pareto']}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"帕累托前沿图已保存为: {filename}")
    plt.show()
    
    # 返回帕累托最优解的索引，方便后续分析
    return pareto_indices

def comprehensive_analysis(schedule, makespan, machines_usage, n_operations, n_machines, best_makespan=None, avg_makespan=None, population=None, objectives=None):
    """
    综合分析调度结果
    
    参数:
        schedule: 操作开始时间数组
        makespan: 总完成时间  
        machines_usage: 机器使用记录
        n_operations: 操作数量
        n_machines: 机器数量
        best_makespan: 每代的最佳makespan值列表（可选）
        avg_makespan: 每代的平均makespan值列表（可选）
        population: 最终种群（可选，用于帕累托分析）
        objectives: 最终种群的目标函数值（可选，用于帕累托分析）
    """
    print("=" * 60)
    print("调度结果综合分析")
    print("=" * 60)
    
    # 基本统计
    total_processing_time = sum(processing_times[(1, op)] for op in range(n_operations))
    total_machine_time = sum(processing_times[(1, op)] * machine_requirements[(1, op)] 
                           for op in range(n_operations))
    
    print(f"总操作数: {n_operations}")
    print(f"总机器数: {n_machines}")
    print(f"总加工时间: {total_processing_time:.2f}")
    print(f"总机器时间: {total_machine_time:.2f}")
    print(f"完工时间 (Makespan): {makespan:.2f}")
    print(f"理论最小完工时间: {total_machine_time / n_machines:.2f}")
    
    # 如果提供了进化过程数据，则绘制进化过程图
    if best_makespan is not None and avg_makespan is not None:
        plot_evolution_process(best_makespan, avg_makespan, "NSGA-III Algorithm Evolution Process")
    
    # 绘制所有图表
    plot_schedule(schedule, makespan, machines_usage, n_operations, n_machines)
    plot_operation_timeline(schedule, n_operations)
    analyze_machine_utilization(machines_usage, makespan, n_machines)
    
    # 如果提供了种群和目标函数值，则绘制帕累托前沿
    if objectives is not None:
        # 找出最佳解的索引
        best_idx = np.argmin(objectives[:, 0])
        
        # 绘制帕累托前沿，突出显示最佳makespan解
        pareto_indices = plot_pareto_front(objectives, highlight_idx=best_idx, 
                                        title="Pareto Front Visualization")
        
        print(f"\n找到 {len(pareto_indices)} 个帕累托最优解")