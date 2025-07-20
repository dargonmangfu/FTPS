import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import processing_times, machine_requirements, precedence_constraints

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
    plt.title("Operation Schedule Gantt Chart", fontsize=16, fontweight='bold')
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
    
    plt.tight_layout()
    plt.savefig('scheduling_gantt.png', dpi=300, bbox_inches='tight')
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
    
    plt.title("Operations Timeline with Precedence Constraints", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Operations (sorted by start time)", fontsize=12)
    plt.yticks(range(len(sorted_ops)), [f'Op{op}' for op in sorted_ops])
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('operation_timeline.png', dpi=300, bbox_inches='tight')
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
    
    plt.tight_layout()
    plt.savefig('machine_utilization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"\n机器利用率统计:")
    print(f"平均利用率: {np.mean(utilization_rates):.3f}")
    print(f"最高利用率: {np.max(utilization_rates):.3f} (Machine {np.argmax(utilization_rates)+1})")
    print(f"最低利用率: {np.min(utilization_rates):.3f} (Machine {np.argmin(utilization_rates)+1})")
    print(f"利用率标准差: {np.std(utilization_rates):.3f}")

def comprehensive_analysis(schedule, makespan, machines_usage, n_operations, n_machines):
    """
    综合分析调度结果
    
    参数:
        schedule: 操作开始时间数组
        makespan: 总完成时间  
        machines_usage: 机器使用记录
        n_operations: 操作数量
        n_machines: 机器数量
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
    
    # 绘制所有图表
    plot_schedule(schedule, makespan, machines_usage, n_operations, n_machines)
    plot_operation_timeline(schedule, n_operations)
    analyze_machine_utilization(machines_usage, makespan, n_machines)