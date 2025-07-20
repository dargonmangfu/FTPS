import numpy as np
import matplotlib.pyplot as plt
from util import uniformpoint, NDsort, lastselection_scheduling
from Modellist import DatasetModel
import copy
import random
from datasets import processing_times, machine_requirements, precedence_constraints
import time
from pyscipopt import Model
# 从evaluate模块导入所有分析函数
from evaluate import plot_schedule, plot_operation_timeline, analyze_machine_utilization, comprehensive_analysis

# 问题参数
n_jobs = 1  # 作业数量
n_operations = 34  # 操作数量(0-33)
n_machines = 10  # 机器数量

# NSGA-III参数
pop_size = 100  # 种群大小
max_gen = 100  # 最大迭代次数
pc = 0.9  # 交叉概率
pm = 0.1  # 变异概率
n_obj = 2  # 目标函数数量

class Solution:
    def __init__(self):
        self.start_times = np.zeros(n_operations)  # 操作开始时间
        self.assigned_machines = np.zeros(n_operations, dtype=int)  # 分配的机器数量
        self.objectives = np.zeros(n_obj)  # 目标函数值
        self.rank = 0  # 非支配等级
        self.is_feasible = False  # 可行性标志
    
    def initialize(self):
        """初始化解决方案"""
        # 设置机器分配为所需的最小值
        self.assigned_machines = np.array([machine_requirements[(1, op)] for op in range(n_operations)])
        
        # 使用拓扑排序正确计算最早开始时间
        earliest_start = np.zeros(n_operations)
        in_degree = np.zeros(n_operations, dtype=int)
        
        # 计算入度
        for op in range(n_operations):
            if op in precedence_constraints:
                in_degree[op] = len(precedence_constraints[op])
        
        # 拓扑排序
        queue = [op for op in range(n_operations) if in_degree[op] == 0]
        
        while queue:
            current_op = queue.pop(0)
            
            # 更新后续操作的入度和最早开始时间
            for next_op in range(n_operations):
                if (next_op in precedence_constraints and 
                    current_op in precedence_constraints[next_op]):
                    earliest_start[next_op] = max(earliest_start[next_op], 
                                                earliest_start[current_op] + processing_times[(1, current_op)])
                    in_degree[next_op] -= 1
                    if in_degree[next_op] == 0:
                        queue.append(next_op)
        
        # 基于正确的最早开始时间设置开始时间
        for op in range(n_operations):
            min_start = earliest_start[op]
            max_start = min_start + processing_times[(1, op)] * 0.5  # 减少随机范围
            self.start_times[op] = random.uniform(min_start, max_start)
        
        # 检查可行性
        self.check_feasibility()
    
    def check_feasibility(self):
        """检查解决方案的可行性"""
        self.is_feasible = True
        for op in range(n_operations):
            if op in precedence_constraints:
                for pred in precedence_constraints[op]:
                    if self.start_times[op] < self.start_times[pred] + processing_times[(1, pred)]:
                        self.is_feasible = False
                        return
        return self.is_feasible

    def decode(self):
        """解码染色体，生成可行调度 - 使用拓扑排序"""
        # 初始化
        actual_start = np.zeros(n_operations)
        machine_available = np.zeros(n_machines)  # 机器可用时间
        operation_machines = {}  # 记录每个操作分配的具体机器
        scheduled = set()  # 已调度的操作
        
        # 计算每个操作的入度（前驱数量）
        in_degree = np.zeros(n_operations, dtype=int)
        for op in range(n_operations):
            if op in precedence_constraints:
                in_degree[op] = len(precedence_constraints[op])
        
        # 找出所有入度为0的操作（可以开始的操作）
        ready_queue = [op for op in range(n_operations) if in_degree[op] == 0]
        
        # 按拓扑顺序调度操作
        while ready_queue:
            # 根据期望开始时间排序ready_queue中的操作
            ready_queue.sort(key=lambda x: self.start_times[x])
            
            # 选择第一个操作进行调度
            current_op = ready_queue.pop(0)
            
            # 计算最早开始时间（考虑前驱约束）
            est = 0
            if current_op in precedence_constraints:
                for pred in precedence_constraints[current_op]:
                    if pred in scheduled:
                        est = max(est, actual_start[pred] + processing_times[(1, pred)])
                    else:
                        # 前驱操作未完成，跳过此操作
                        ready_queue.append(current_op)
                        continue
            
            # 分配机器
            required = machine_requirements[(1, current_op)]
            sorted_machines = np.argsort(machine_available)
            assigned_machines = sorted_machines[:required]
            
            # 计算开始时间
            machine_est = max(machine_available[m] for m in assigned_machines) if len(assigned_machines) > 0 else 0
            start_time = max(est, machine_est)
            
            # 更新机器可用时间
            for m in assigned_machines:
                machine_available[m] = start_time + processing_times[(1, current_op)]
            
            actual_start[current_op] = start_time
            operation_machines[current_op] = assigned_machines
            scheduled.add(current_op)
            
            # 更新后续操作的入度，将新的可调度操作加入ready_queue
            for next_op in range(n_operations):
                if (next_op not in scheduled and 
                    next_op in precedence_constraints and 
                    current_op in precedence_constraints[next_op]):
                    in_degree[next_op] -= 1
                    if in_degree[next_op] == 0 and next_op not in ready_queue:
                        ready_queue.append(next_op)
        
        # 验证是否所有操作都被调度
        if len(scheduled) != n_operations:
            print(f"警告: 只有 {len(scheduled)} 个操作被调度，总共有 {n_operations} 个操作")
            # 强制调度未调度的操作
            unscheduled = [op for op in range(n_operations) if op not in scheduled]
            for op in unscheduled:
                required = machine_requirements[(1, op)]
                sorted_machines = np.argsort(machine_available)
                assigned_machines = sorted_machines[:required]
                machine_est = max(machine_available[m] for m in assigned_machines) if len(assigned_machines) > 0 else 0
                actual_start[op] = machine_est
                for m in assigned_machines:
                    machine_available[m] = machine_est + processing_times[(1, op)]
                operation_machines[op] = assigned_machines
        
        # 计算目标函数
        makespan = max(actual_start[i] + processing_times[(1, i)] for i in range(n_operations))
        total_busy_time = sum(processing_times[(1, i)] * machine_requirements[(1, i)] for i in range(n_operations))
        max_possible_busy_time = makespan * n_machines
        resource_utilization = total_busy_time / max_possible_busy_time if max_possible_busy_time > 0 else 1.0
        
        return makespan, 1 - resource_utilization, actual_start, operation_machines

    def evaluate(self):
        """评估解决方案"""
        try:
            makespan, resource_utilization, _, _ = self.decode()
            self.objectives[0] = makespan
            self.objectives[1] = resource_utilization
            self.check_feasibility()
            
            # 降低惩罚因子，避免过度惩罚
            if not self.is_feasible:
                penalty_factor = 1.1
                self.objectives[0] *= penalty_factor
                self.objectives[1] *= penalty_factor
        except Exception as e:
            print(f"评估错误: {e}")
            self.objectives = [float('inf'), float('inf')]

# 初始化种群
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        solution = Solution()
        solution.initialize()
        solution.evaluate()
        population.append(solution)
    return population

# 交叉操作
def crossover(parent1, parent2):
    child = Solution()
    crossover_point = random.randint(1, n_operations-1)
    
    # 组合父代的开始时间
    child.start_times = np.concatenate([
        parent1.start_times[:crossover_point],
        parent2.start_times[crossover_point:]
    ])
    
    # 组合父代的机器分配
    child.assigned_machines = np.concatenate([
        parent1.assigned_machines[:crossover_point],
        parent2.assigned_machines[crossover_point:]
    ])
    
    # 确保机器分配有效
    for i in range(n_operations):
        req = machine_requirements[(1, i)]
        child.assigned_machines[i] = max(1, min(req, child.assigned_machines[i]))
    
    return child

# 变异操作
def mutation(solution):
    mutated = copy.deepcopy(solution)
    for i in range(n_operations):
        if random.random() < pm:
            # 减小变异范围
            mutation_range = processing_times[(1, i)] * 0.3
            mutated.start_times[i] = max(0, mutated.start_times[i] + random.uniform(-mutation_range, mutation_range))
            
            # 变异机器分配
            req = machine_requirements[(1, i)]
            if req > 1 and random.random() < 0.2:
                mutated.assigned_machines[i] = random.randint(1, req)
    
    return mutated

# NSGA-III选择
def nsga3_selection(population, ref_points):
    objectives = np.array([ind.objectives for ind in population])
    zmin = np.min(objectives, axis=0)
    ranks, max_rank = NDsort(objectives, pop_size, n_obj)
    
    next_population = []
    for rank in range(1, max_rank + 1):
        current_front = [j for j, r in enumerate(ranks) if r == rank]
        
        if len(next_population) + len(current_front) <= pop_size:
            next_population.extend([population[j] for j in current_front])
        else:
            # 需要从当前前沿选择个体
            k = pop_size - len(next_population)
            
            if not next_population:  # 如果next_population为空
                # 直接选择前k个个体，避免维度不匹配问题
                chosen_indices = current_front[:k]
            else:
                try:
                    selected_obj = np.array([ind.objectives for ind in next_population])
                    front_obj = objectives[current_front]
                    
                    # 检查数组维度
                    if selected_obj.shape[1] != front_obj.shape[1]:
                        print(f"警告: 数组维度不匹配! selected_obj:{selected_obj.shape}, front_obj:{front_obj.shape}")
                        # 直接选择前k个个体作为备选方案
                        chosen_indices = current_front[:k]
                    else:
                        chosen = lastselection_scheduling(selected_obj, front_obj, k, ref_points, zmin)
                        chosen_indices = [current_front[j] for j, is_chosen in enumerate(chosen) if is_chosen]
                except Exception as e:
                    print(f"选择过程中出错: {e}")
                    # 发生错误时的备选方案
                    chosen_indices = current_front[:k]
            
            next_population.extend([population[j] for j in chosen_indices])
            break
    
    return next_population

# NSGA-III主算法
def nsga3_for_scheduling():
    start_time = time.time()
    population = initialize_population(pop_size)
    ref_points, _ = uniformpoint(pop_size, n_obj)
    
    # 记录进化过程
    best_makespan_history = []
    best_utilization_history = []
    
    for gen in range(max_gen):
        # 创建子代
        offspring = []
        for _ in range(pop_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutation(child)
            child.evaluate()
            offspring.append(child)
        
        # 合并种群
        combined = population + offspring
        
        # 环境选择
        population = nsga3_selection(combined, ref_points)
        
        # 记录最佳值
        best_makespan = min(ind.objectives[0] for ind in population)
        best_utilization = min(ind.objectives[1] for ind in population)
        best_makespan_history.append(best_makespan)
        best_utilization_history.append(best_utilization)
        
        print(f"Generation {gen+1}: Makespan={best_makespan:.2f}, Utilization={best_utilization:.4f}")
    
    # 输出结果
    print(f"\nTotal time: {time.time()-start_time:.2f} seconds")
    
    # 绘制进化过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(best_makespan_history)
    plt.title("Makespan Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Makespan")
    
    plt.subplot(1, 2, 2)
    plt.plot(best_utilization_history)
    plt.title("Resource Utilization Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Resource Utilization")
    
    plt.tight_layout()
    plt.savefig("evolution.png")
    plt.show()
    
    # 返回最佳解和最终种群
    best_solution = min(population, key=lambda x: x.objectives[0])
    return best_solution, population


# 获取机器使用详情的函数
def get_machine_usage(actual_start, operation_machines):
    """
    获取机器使用详情，确保每个操作在分配的机器上连续完成
    
    参数:
        actual_start: 操作实际开始时间
        operation_machines: 每个操作分配的具体机器字典
    
    返回:
        列表 [(机器ID, 开始时间, 结束时间, 操作ID), ...]
    """
    machines_usage = []
    
    # 为每个操作在其分配的机器上创建记录
    for op in range(n_operations):
        start_time = actual_start[op]
        duration = processing_times[(1, op)]
        end_time = start_time + duration
        assigned_machines = operation_machines[op]
        
        # 每个分配的机器都要在整个操作期间被占用
        for machine_id in assigned_machines:
            machines_usage.append((machine_id, start_time, end_time, op))
    
    return machines_usage

# 修改主函数以调用evaluate模块中的函数
if __name__ == "__main__":
    best_solution, final_population = nsga3_for_scheduling()
    makespan, utilization, schedule, operation_machines = best_solution.decode()
    
    print("\nOptimal Schedule (按拓扑顺序):")
    # 按实际开始时间排序显示
    sorted_ops = sorted(range(n_operations), key=lambda x: schedule[x])
    for op in sorted_ops:
        predecessors = precedence_constraints.get(op, [])
        print(f"Operation {op}: Start={schedule[op]:.1f}, "
              f"Duration={processing_times[(1, op)]}, "
              f"Machines={machine_requirements[(1, op)]}, "
              f"Assigned to: {operation_machines[op]}, "
              f"Predecessors: {predecessors}")
    
    # 验证前驱约束
    print("\n前驱约束验证:")
    constraint_violations = 0
    for op in range(n_operations):
        if op in precedence_constraints:
            for pred in precedence_constraints[op]:
                pred_finish = schedule[pred] + processing_times[(1, pred)]
                if schedule[op] < pred_finish:
                    print(f"约束违反: Op{op} 开始时间 {schedule[op]:.1f} < Op{pred} 完成时间 {pred_finish:.1f}")
                    constraint_violations += 1
    
    if constraint_violations == 0:
        print("所有前驱约束都得到满足！")
    else:
        print(f"发现 {constraint_violations} 个约束违反！")
    
    print(f"\nFinal Makespan: {makespan:.2f}")
    print(f"Resource Utilization: {1 - utilization:.4f}")
    
    # 获取机器使用详情
    machines_usage = get_machine_usage(schedule, operation_machines)
    
    # 使用evaluate模块的综合分析功能
    comprehensive_analysis(schedule, makespan, machines_usage, n_operations, n_machines)
    
    # 绘制帕累托前沿图
    plt.figure(figsize=(10, 6))
    objectives = np.array([ind.objectives for ind in final_population])
    plt.scatter(objectives[:, 0], objectives[:, 1], c='blue', alpha=0.7)
    plt.title("Pareto Front")
    plt.xlabel("Makespan")
    plt.ylabel("Resource Utilization")
    plt.grid(True)
    plt.savefig('pareto_front.png')
    plt.show()
    plt.show()
