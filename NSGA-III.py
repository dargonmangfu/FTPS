import numpy as np
import matplotlib.pyplot as plt
from util import uniformpoint, NDsort, lastselection_scheduling
import copy
import random
from datasets import processing_times, machine_requirements, precedence_constraints
import time
from pyscipopt import Model
# 从evaluate模块导入所有分析函数
from evaluate import comprehensive_analysis

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
        # 改用优先级编码而非直接的开始时间
        self.priorities = np.zeros(n_operations)  # 操作优先级
        self.machine_preferences = np.zeros(n_operations, dtype=int)  # 机器偏好
        self.start_times = np.zeros(n_operations)  # 实际开始时间（解码后得到）
        self.assigned_machines = np.zeros(n_operations, dtype=int)  # 分配的机器数量
        self.objectives = np.zeros(n_obj)  # 目标函数值
        self.rank = 0  # 非支配等级
        self.is_feasible = False  # 可行性标志
    
    def initialize(self):
        """初始化解决方案 - 使用优先级编码"""
        # 随机生成操作优先级
        self.priorities = np.random.rand(n_operations)
        
        # 设置机器偏好（在允许范围内随机选择）
        for op in range(n_operations):
            required = machine_requirements[(1, op)]
            # 机器偏好设置为需要的机器数量（简化处理）
            self.machine_preferences[op] = required
        
        # 解码生成实际调度
        self.decode_and_evaluate()
    
    def decode_and_evaluate(self):
        """解码染色体并评估"""
        self.decode()
        self.evaluate()
    
    def decode(self):
        """解码染色体，生成可行调度"""
        # 初始化
        actual_start = np.zeros(n_operations)
        machine_busy_until = np.zeros(n_machines)  # 每台机器的忙碌结束时间
        operation_machines = {}  # 记录每个操作分配的具体机器
        
        # 根据拓扑约束和优先级确定调度顺序
        scheduled = set()
        schedule_order = []
        
        # 构建拓扑排序的调度序列
        while len(scheduled) < n_operations:
            candidates = []
            
            # 找出所有前驱已完成的操作
            for op in range(n_operations):
                if op not in scheduled:
                    if op not in precedence_constraints:
                        candidates.append(op)
                    else:
                        # 检查所有前驱是否已调度
                        all_pred_scheduled = all(pred in scheduled 
                                               for pred in precedence_constraints[op])
                        if all_pred_scheduled:
                            candidates.append(op)
            
            if not candidates:
                # 如果没有候选操作，可能存在循环依赖
                remaining = [op for op in range(n_operations) if op not in scheduled]
                print(f"警告：可能存在循环依赖，强制调度剩余操作: {remaining}")
                candidates = remaining[:1]  # 选择一个操作强制调度
            
            # 根据优先级选择下一个操作
            next_op = max(candidates, key=lambda x: self.priorities[x])
            schedule_order.append(next_op)
            scheduled.add(next_op)
        
        # 按确定的顺序调度操作
        for op in schedule_order:
            # 计算最早开始时间（考虑前驱约束）
            est_due_to_precedence = 0
            if op in precedence_constraints:
                for pred in precedence_constraints[op]:
                    pred_finish = actual_start[pred] + processing_times[(1, pred)]
                    est_due_to_precedence = max(est_due_to_precedence, pred_finish)
            
            # 分配机器
            required_machines = machine_requirements[(1, op)]
            
            # 选择最早可用的机器组合
            machine_candidates = list(range(n_machines))
            machine_candidates.sort(key=lambda x: machine_busy_until[x])
            
            assigned_machines_list = machine_candidates[:required_machines]
            
            # 计算考虑机器可用性的最早开始时间
            est_due_to_machines = max(machine_busy_until[m] for m in assigned_machines_list)
            
            # 最终开始时间
            start_time = max(est_due_to_precedence, est_due_to_machines)
            
            # 更新机器忙碌时间
            duration = processing_times[(1, op)]
            for m in assigned_machines_list:
                machine_busy_until[m] = start_time + duration
            
            # 记录结果
            actual_start[op] = start_time
            operation_machines[op] = assigned_machines_list
            self.assigned_machines[op] = required_machines
        
        # 更新实例变量
        self.start_times = actual_start
        self.operation_machines = operation_machines
        
        # 检查可行性
        self.check_feasibility()
        
        return actual_start, operation_machines
    
    def check_feasibility(self):
        """检查解决方案的可行性"""
        self.is_feasible = True
        
        # 检查前驱约束
        for op in range(n_operations):
            if op in precedence_constraints:
                for pred in precedence_constraints[op]:
                    pred_finish = self.start_times[pred] + processing_times[(1, pred)]
                    if self.start_times[op] < pred_finish - 1e-6:  # 允许小的数值误差
                        self.is_feasible = False
                        return False
        
        # 检查机器冲突
        machine_schedules = [[] for _ in range(n_machines)]
        for op in range(n_operations):
            start = self.start_times[op]
            end = start + processing_times[(1, op)]
            for machine_id in self.operation_machines[op]:
                machine_schedules[machine_id].append((start, end, op))
        
        # 检查每台机器是否有时间冲突
        for machine_id in range(n_machines):
            schedule = sorted(machine_schedules[machine_id])
            for i in range(len(schedule) - 1):
                if schedule[i][1] > schedule[i+1][0] + 1e-6:  # 允许小的数值误差
                    self.is_feasible = False
                    return False
        
        return True

    def evaluate(self):
        """评估解决方案"""
        try:
            # 计算makespan
            makespan = max(self.start_times[i] + processing_times[(1, i)] 
                          for i in range(n_operations))
            
            # 计算资源利用率
            total_busy_time = sum(processing_times[(1, i)] * machine_requirements[(1, i)] 
                                 for i in range(n_operations))
            max_possible_busy_time = makespan * n_machines
            resource_utilization = total_busy_time / max_possible_busy_time if max_possible_busy_time > 0 else 1.0
            
            self.objectives[0] = makespan
            self.objectives[1] = 1 - resource_utilization  # 最小化资源利用率的倒数
            
            # 对不可行解施加惩罚
            if not self.is_feasible:
                penalty_factor = 2.0  # 增加惩罚力度
                self.objectives[0] *= penalty_factor
                self.objectives[1] *= penalty_factor
                
        except Exception as e:
            print(f"评估错误: {e}")
            self.objectives = [float('inf'), float('inf')]
            self.is_feasible = False

# 初始化种群
def initialize_population(pop_size):
    population = []
    feasible_count = 0
    
    for i in range(pop_size):
        solution = Solution()
        solution.initialize()
        if solution.is_feasible:
            feasible_count += 1
        population.append(solution)
    
    print(f"初始种群中可行解数量: {feasible_count}/{pop_size}")
    return population

# 改进的交叉操作
def crossover(parent1, parent2):
    """基于优先级的交叉操作"""
    child = Solution()
    
    # 对优先级进行交叉
    crossover_point = random.randint(1, n_operations-1)
    child.priorities = np.concatenate([
        parent1.priorities[:crossover_point],
        parent2.priorities[crossover_point:]
    ])
    
    # 对机器偏好进行交叉
    child.machine_preferences = np.concatenate([
        parent1.machine_preferences[:crossover_point],
        parent2.machine_preferences[crossover_point:]
    ])
    
    # 确保机器偏好在有效范围内
    for i in range(n_operations):
        req = machine_requirements[(1, i)]
        child.machine_preferences[i] = max(1, min(req, child.machine_preferences[i]))
    
    return child

# 改进的变异操作
def mutation(solution):
    """基于优先级的变异操作"""
    mutated = copy.deepcopy(solution)
    
    for i in range(n_operations):
        if random.random() < pm:
            # 变异优先级
            mutated.priorities[i] = random.random()
            
            # 小概率变异机器偏好
            if random.random() < 0.3:
                req = machine_requirements[(1, i)]
                if req > 1:
                    mutated.machine_preferences[i] = random.randint(1, req)
    
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
    feasible_count_history = []
    
    for gen in range(max_gen):
        # 创建子代
        offspring = []
        for _ in range(pop_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutation(child)
            child.decode_and_evaluate()  # 使用新的解码评估方法
            offspring.append(child)
        
        # 合并种群
        combined = population + offspring
        
        # 环境选择
        population = nsga3_selection(combined, ref_points)
        
        # 统计可行解数量
        feasible_count = sum(1 for ind in population if ind.is_feasible)
        feasible_count_history.append(feasible_count)
        
        # 记录最佳值（只考虑可行解）
        feasible_solutions = [ind for ind in population if ind.is_feasible]
        if feasible_solutions:
            best_makespan = min(ind.objectives[0] for ind in feasible_solutions)
            best_utilization = min(ind.objectives[1] for ind in feasible_solutions)
        else:
            best_makespan = min(ind.objectives[0] for ind in population)
            best_utilization = min(ind.objectives[1] for ind in population)
        
        best_makespan_history.append(best_makespan)
        best_utilization_history.append(best_utilization)
        
        print(f"Generation {gen+1}: Makespan={best_makespan:.2f}, "
              f"Utilization={best_utilization:.4f}, Feasible={feasible_count}/{pop_size}")
    
    # 输出结果
    print(f"\nTotal time: {time.time()-start_time:.2f} seconds")
    
    # 绘制进化过程
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(best_makespan_history)
    plt.title("Makespan Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Makespan")
    
    plt.subplot(1, 3, 2)
    plt.plot(best_utilization_history)
    plt.title("Resource Utilization Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Resource Utilization")
    
    plt.subplot(1, 3, 3)
    plt.plot(feasible_count_history)
    plt.title("Feasible Solutions Count")
    plt.xlabel("Generation")
    plt.ylabel("Feasible Count")
    
    plt.tight_layout()
    plt.savefig("evolution.png")
    plt.show()
    
    # 返回最佳可行解
    feasible_solutions = [ind for ind in population if ind.is_feasible]
    if feasible_solutions:
        best_solution = min(feasible_solutions, key=lambda x: x.objectives[0])
    else:
        best_solution = min(population, key=lambda x: x.objectives[0])
        print("警告：没有找到可行解，返回最佳不可行解")
    
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

# 修改主函数
if __name__ == "__main__":
    best_solution, final_population = nsga3_for_scheduling()
    
    print(f"\n最佳解是否可行: {best_solution.is_feasible}")
    
    # 获取调度结果
    schedule = best_solution.start_times
    operation_machines = best_solution.operation_machines
    
    print("\nOptimal Schedule (按开始时间排序):")
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
                if schedule[op] < pred_finish - 1e-6:
                    print(f"约束违反: Op{op} 开始时间 {schedule[op]:.1f} < Op{pred} 完成时间 {pred_finish:.1f}")
                    constraint_violations += 1
    
    if constraint_violations == 0:
        print("所有前驱约束都得到满足！")
    else:
        print(f"发现 {constraint_violations} 个约束违反！")
    
    makespan = max(schedule[i] + processing_times[(1, i)] for i in range(n_operations))
    total_busy_time = sum(processing_times[(1, i)] * machine_requirements[(1, i)] for i in range(n_operations))
    resource_utilization = total_busy_time / (makespan * n_machines)
    
    print(f"\nFinal Makespan: {makespan:.2f}")
    print(f"Resource Utilization: {resource_utilization:.4f}")
    
    # 获取机器使用详情
    machines_usage = get_machine_usage(schedule, operation_machines)
    
    # 使用evaluate模块的综合分析功能
    comprehensive_analysis(schedule, makespan, machines_usage, n_operations, n_machines)
    
    # 绘制帕累托前沿图
    plt.figure(figsize=(10, 6))
    objectives = np.array([ind.objectives for ind in final_population])
    feasible_obj = np.array([ind.objectives for ind in final_population if ind.is_feasible])
    
    plt.scatter(objectives[:, 0], objectives[:, 1], c='red', alpha=0.3, label='Infeasible')
    if len(feasible_obj) > 0:
        plt.scatter(feasible_obj[:, 0], feasible_obj[:, 1], c='blue', alpha=0.7, label='Feasible')
    
    plt.title("Pareto Front")
    plt.xlabel("Makespan")
    plt.ylabel("Resource Utilization")
    plt.legend()
    plt.grid(True)
    plt.savefig('pareto_front.png')
    plt.show()
