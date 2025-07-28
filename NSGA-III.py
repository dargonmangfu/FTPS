import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from util import uniformpoint, NDsort, lastselection_scheduling,GO
from datasets import get_model_data, processing_times, machine_requirements, precedence_constraints, mutual_exclusion_constraints
from evaluate import comprehensive_analysis
import os
import datetime

class SchedulingNSGAIII:
    def __init__(self, pop_size=100, max_gen=100, pc=0.9, pm=0.1, t1=20, t2=20):
        """
        初始化NSGA-III算法参数
        
        参数:
            pop_size: 种群大小
            max_gen: 最大迭代次数
            pc: 交叉概率
            pm: 变异概率
            t1: 模拟二进制交叉(SBX)的分布指数
            t2: 多项式变异的分布指数
        """
        # 获取问题数据
        self.n, self.jobs_operations, self.proc_times, self.machine_reqs, self.precedence, self.mutual_exclusion = get_model_data()
        
        # 初始化算法参数
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pc = pc
        self.pm = pm
        self.t1 = t1
        self.t2 = t2
        
        # 初始化问题特定参数
        self.n_operations = 34  # 操作总数(0-33)
        self.n_machines = 10    # 机器总数
        
        # 多目标
        self.M = 2  # 目标函数个数: 1.完工时间 2.资源利用不平衡性
        
        # 计算参考点
        self.Z, self.N = uniformpoint(self.pop_size, self.M)

    def initialize_population(self):
        """初始化种群"""
        # 使用优先级列表表示
        population = []
        for _ in range(self.pop_size):
            # 生成随机优先级值
            priorities = np.random.random(self.n_operations)
            chromosome = priorities
            population.append(chromosome)
        
        return np.array(population)
    
    def decode_chromosome(self, chromosome):
        """
        将染色体解码为可行的调度方案
        
        参数:
            chromosome: 表示操作优先级的数组
        
        返回:
            schedule: 每个操作的开始时间数组
            makespan: 总完工时间
            machines_usage: 机器使用记录
        """
        # 初始化调度数据
        schedule = np.zeros(self.n_operations)  # 操作的开始时间
        finish_times = np.zeros(self.n_operations)  # 操作的完成时间
        machine_avail_time = np.zeros(self.n_machines)  # 机器的可用时间
        
        # 追踪机器的使用情况
        machines_usage = []  # 记录 (机器ID, 开始时间, 结束时间, 操作ID)
        
        # 创建未完成操作的列表
        unscheduled = list(range(self.n_operations))
        
        # 创建已经满足前驱约束的操作列表
        ready = []
        for op in unscheduled:
            if op not in precedence_constraints or not precedence_constraints[op]:
                ready.append(op)
        
        # 主调度循环
        while unscheduled:
            # 如果ready列表为空，强制添加一个可调度的操作
            if not ready:
                # 找到所有前驱已完成的操作
                for op in unscheduled:
                    if op in precedence_constraints:
                        if all(pred not in unscheduled for pred in precedence_constraints[op]):
                            ready.append(op)
                    else:
                        ready.append(op)
                
                # 如果还是没有可调度的操作，说明存在循环依赖，强制添加第一个
                if not ready:
                    ready.append(unscheduled[0])
            
            # 按照染色体中的优先级值对ready列表排序
            ready.sort(key=lambda x: -chromosome[x])  # 值越大优先级越高
            
            # 尝试调度ready列表中的操作
            scheduled_this_round = False
            i = 0
            
            while i < len(ready) and not scheduled_this_round:
                current_op = ready[i]
                
                # 检查是否可以调度（考虑交叉关系约束）
                can_schedule = True
                
                if current_op in mutual_exclusion_constraints:
                    conflicting_op = mutual_exclusion_constraints[current_op]
                    # 如果冲突操作已经调度但还未完成，检查时间冲突
                    if conflicting_op not in unscheduled:
                        # 计算当前操作的最早开始时间
                        earliest_start = 0
                        
                        # 考虑前驱约束
                        if current_op in precedence_constraints:
                            for pred in precedence_constraints[current_op]:
                                if finish_times[pred] > earliest_start:
                                    earliest_start = finish_times[pred]
                        
                        # 如果冲突操作还在执行中，需要等待其完成
                        if finish_times[conflicting_op] > earliest_start:
                            earliest_start = finish_times[conflicting_op]
                
                if can_schedule:
                    # 调度当前操作
                    ready.pop(i)
                    unscheduled.remove(current_op)
                    
                    # 获取此操作需要的机器数量
                    req_machines = machine_requirements[(1, current_op)]
                    
                    # 找到最早可以开始的时间
                    earliest_start = 0
                    
                    # 考虑前驱约束
                    if current_op in precedence_constraints:
                        for pred in precedence_constraints[current_op]:
                            if finish_times[pred] > earliest_start:
                                earliest_start = finish_times[pred]
                    
                    # 考虑交叉关系约束
                    if current_op in mutual_exclusion_constraints:
                        conflicting_op = mutual_exclusion_constraints[current_op]
                        if conflicting_op not in unscheduled and finish_times[conflicting_op] > 0:
                            earliest_start = max(earliest_start, finish_times[conflicting_op])
                    
                    # 分配操作到机器上（考虑机器可用性）
                    available_machines = sorted(range(self.n_machines), key=lambda m: machine_avail_time[m])
                    
                    # 获取所需数量的机器
                    assigned_machines = available_machines[:req_machines]
                    
                    # 找到这些机器中最早的公共可用时间
                    start_time = max(earliest_start, max(machine_avail_time[m] for m in assigned_machines))
                    
                    # 更新操作的开始时间
                    schedule[current_op] = start_time
                    
                    # 计算操作的完成时间
                    end_time = start_time + processing_times[(1, current_op)]
                    finish_times[current_op] = end_time
                    
                    # 更新机器的可用时间
                    for m in assigned_machines:
                        machine_avail_time[m] = end_time
                        machines_usage.append((m, start_time, end_time, current_op))
                    
                    scheduled_this_round = True
                    
                    # 检查新的可执行操作
                    for op in unscheduled:
                        if op not in ready:
                            if op in precedence_constraints:
                                if all(pred not in unscheduled for pred in precedence_constraints[op]):
                                    ready.append(op)
                            else:
                                ready.append(op)
                else:
                    i += 1
            
            # 如果这一轮没有调度任何操作，强制调度第一个ready操作
            if not scheduled_this_round and ready:
                current_op = ready.pop(0)
                unscheduled.remove(current_op)
                
                # 强制调度
                req_machines = machine_requirements[(1, current_op)]
                available_machines = sorted(range(self.n_machines), key=lambda m: machine_avail_time[m])
                assigned_machines = available_machines[:req_machines]
                
                earliest_start = 0
                if current_op in precedence_constraints:
                    for pred in precedence_constraints[current_op]:
                        if finish_times[pred] > earliest_start:
                            earliest_start = finish_times[pred]
                
                start_time = max(earliest_start, max(machine_avail_time[m] for m in assigned_machines))
                schedule[current_op] = start_time
                end_time = start_time + processing_times[(1, current_op)]
                finish_times[current_op] = end_time
                
                for m in assigned_machines:
                    machine_avail_time[m] = end_time
                    machines_usage.append((m, start_time, end_time, current_op))
        
        # 计算总完工时间
        makespan = max(finish_times)
        
        return schedule, makespan, machines_usage
    
    def evaluate_population(self, population):
        """评估种群中的每个个体"""
        objectives = np.zeros((len(population), self.M))
        
        for i, chrom in enumerate(population):
            schedule, makespan, machines_usage = self.decode_chromosome(chrom)
            
            # 目标1: 最小化完工时间
            objectives[i, 0] = makespan
            
            # 目标2: 最小化资源利用不平衡
            machine_busy_time = np.zeros(self.n_machines)
            for m, start, end, _ in machines_usage:
                machine_busy_time[m] += (end - start)
            
            # 计算不平衡指标 (使用标准差)
            imbalance = np.std(machine_busy_time / makespan)
            objectives[i, 1] = imbalance
        
        return objectives
    
    def roulette_wheel_selection(self, population, objectives):
        """轮盘赌选择（基于适应度排名）"""
        selected = []
        n_pop = len(population)
        
        # 计算非支配排序
        fronts, _ = NDsort(objectives, n_pop, self.M)
        
        # 转换为适应度值（前沿越小适应度越高）
        max_front = np.max(fronts)
        fitness = max_front + 1 - fronts
        
        # 归一化适应度
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness
        
        # 选择个体
        for _ in range(n_pop):
            selected_idx = np.random.choice(n_pop, p=probabilities)
            selected.append(population[selected_idx])
        
        return np.array(selected)
    
    def environment_selection(self, population, offspring):
        """环境选择操作"""
        # 合并父代和子代
        combined_pop = np.vstack([population, offspring])
        
        # 评估合并后的种群
        combined_obj = self.evaluate_population(combined_pop)
        
        # 非支配排序
        fronts, max_front = NDsort(combined_obj, self.pop_size, self.M)
        
        # 选择下一代种群
        next_pop_indices = fronts < max_front
        
        # 对最后一个前沿进行选择
        last_front = np.where(fronts == max_front)[0]
        
        # 计算理想点
        z_min = np.min(combined_obj, axis=0)
        
        # 如果需要从最后一个前沿中选择个体
        if sum(next_pop_indices) < self.pop_size:
            # 需要从最后一个前沿中选择的个体数量
            k = self.pop_size - sum(next_pop_indices)
            
            # 选择个体
            selected = lastselection_scheduling(
                combined_obj[next_pop_indices], 
                combined_obj[last_front],
                k, 
                self.Z, 
                z_min
            )
            
            # 更新选择的个体
            next_pop_indices[last_front[selected]] = True
        
        # 返回选择的个体
        return combined_pop[next_pop_indices]
    
    def run(self):
        """运行NSGA-III算法"""
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        objectives = self.evaluate_population(population)
        
        # 记录每代的最佳makespan
        best_makespan = []
        avg_makespan = []
        
        # 开始进化
        for gen in range(self.max_gen):
            # 检查种群多样性
            # if gen % 20 == 0:
            #    unique_count = len(np.unique(population.round(4), axis=0))
            #    print(f"第 {gen} 代: 唯一个体数 = {unique_count}/{len(population)}")
            
            # 使用轮盘赌选择
            mating_pool = self.roulette_wheel_selection(population, objectives)
            
            # 交叉和变异
            offspring = GO(mating_pool, self.t1, self.t2, self.pc, self.pm)
            
            # 环境选择
            population = self.environment_selection(population, offspring)
            
            # 评估新种群
            objectives = self.evaluate_population(population)
            
            # 记录当前代最佳makespan和平均makespan
            min_makespan = np.min(objectives[:, 0])
            mean_makespan = np.mean(objectives[:, 0])
            best_makespan.append(min_makespan)
            avg_makespan.append(mean_makespan)
            
            # 输出进度
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}/{self.max_gen}, Best Makespan: {min_makespan:.2f}, Avg Makespan: {mean_makespan:.2f}")
        
        # 找到最佳解的索引 (基于makespan)
        best_idx = np.argmin(objectives[:, 0])
        best_solution = population[best_idx]
        
        # 解码最佳解
        schedule, makespan, machines_usage = self.decode_chromosome(best_solution)
        
        # 修改返回值，增加最终种群和目标值
        return schedule, makespan, machines_usage, best_solution, objectives[best_idx], best_makespan, avg_makespan, population, objectives

# 运行算法
if __name__ == "__main__":
    # 移除固定随机种子，使每次运行结果不同
    # np.random.seed(42)
    
    # 可以选择使用当前时间作为随机种子，每次运行都不同
    import time
    current_seed = int(time.time()) % 10000
    np.random.seed(current_seed)
    print(f"使用随机种子: {current_seed}")
    
    # 允许通过命令行参数配置算法参数
    import sys
    pop_size = 200
    max_gen = 200
    pc = 0.9  # 交叉概率
    pm = 0.2  # 变异概率
    
    # 可以通过命令行参数修改这些值
    if len(sys.argv) > 1:
        try:
            pop_size = int(sys.argv[1])
            if len(sys.argv) > 2:
                max_gen = int(sys.argv[2])
            if len(sys.argv) > 3:
                pc = float(sys.argv[3])
            if len(sys.argv) > 4:
                pm = float(sys.argv[4])
        except:
            print("参数格式错误，使用默认值")
    
    print(f"算法参数: 种群大小={pop_size}, 最大代数={max_gen}, 交叉概率={pc}, 变异概率={pm}")
    
    # 创建NSGA-III实例
    nsga3 = SchedulingNSGAIII(pop_size=pop_size, max_gen=max_gen, pc=pc, pm=pm)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行算法并获取所有返回值，包括进化记录和最终种群
        schedule, makespan, machines_usage, best_solution, best_objectives, best_makespan_history, avg_makespan_history, population, objectives = nsga3.run()
        
        # 记录结束时间
        end_time = time.time()
        
        # 输出结果
        print("\n" + "="*50)
        print("最优调度结果:")
        print(f"完工时间 (Makespan): {makespan:.2f}")
        print(f"资源不平衡指标: {best_objectives[1]:.4f}")
        print(f"算法运行时间: {end_time - start_time:.2f} 秒")
        
        # 显示详细的调度结果
        print("\n操作开始时间:")
        for op in range(nsga3.n_operations):
            print(f"操作 {op}: {schedule[op]:.2f}")
        
        # 使用evaluate.py中的函数进行综合分析，传入进化过程数据和最终种群
        comprehensive_analysis(schedule, makespan, machines_usage, nsga3.n_operations, nsga3.n_machines, 
                              best_makespan=best_makespan_history, avg_makespan=avg_makespan_history,
                              population=population, objectives=objectives)

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 输出结果
        print("\n" + "="*50)
        print("最优调度结果:")
        print(f"完工时间 (Makespan): {makespan:.2f}")
        print(f"资源不平衡指标: {best_objectives[1]:.4f}")
        print(f"算法运行时间: {end_time - start_time:.2f} 秒")
        
        # 显示详细的调度结果
        print("\n操作开始时间:")
        for op in range(nsga3.n_operations):
            print(f"操作 {op}: {schedule[op]:.2f}")
        
        # 使用evaluate.py中的函数进行综合分析，传入进化过程数据和最终种群
        comprehensive_analysis(schedule, makespan, machines_usage, nsga3.n_operations, nsga3.n_machines, 
                              best_makespan=best_makespan_history, avg_makespan=avg_makespan_history,
                              population=population, objectives=objectives)

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
