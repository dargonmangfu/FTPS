processing_times = {(1, 0): 360, (1, 1): 360, (1, 2): 300, (1, 3): 180, (1, 4): 180, (1, 5): 240,
                    (1, 6): 1200, (1, 7): 180, (1, 8): 15, (1, 9): 100, (1, 10): 300, (1, 11): 600,
                    (1, 12): 120, (1, 13): 120, (1, 14): 120, (1, 15): 720, (1, 16): 120, (1, 17): 300,
                    (1, 18): 290, (1, 19): 120, (1, 20): 120, (1, 21): 120, (1, 22): 240, (1, 23): 900,
                    (1, 24): 240, (1, 25): 900, (1, 26): 360, (1, 27): 180, (1, 28): 600, (1, 29): 360,
                    (1, 30): 180, (1, 31): 180, (1, 32): 170, (1, 33): 60}

machine_requirements = {(1, 0): 4, (1, 1): 4, (1, 2): 2, (1, 3): 1, (1, 4): 1, (1, 5): 2,
                    (1, 6): 1, (1, 7): 4, (1, 8): 4, (1, 9): 4, (1, 10): 4, (1, 11): 2,
                    (1, 12): 4, (1, 13): 4, (1, 14): 2, (1, 15): 2, (1, 16): 2, (1, 17): 2,
                    (1, 18): 1, (1, 19): 4, (1, 20): 6, (1, 21): 6, (1, 22): 2, (1, 23): 2,
                    (1, 24): 2, (1, 25): 2, (1, 26): 2, (1, 27): 3, (1, 28): 2, (1, 29): 2,
                    (1, 30): 1, (1, 31): 4, (1, 32): 4, (1, 33): 2}

precedence_constraints = {0: [], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [1, 2], 8: [1, 2], 9: [7, 8],
                          10: [7, 8], 11: [7, 8], 12: [3, 9, 10, 11], 13: [9, 10, 11], 14: [4, 12], 15: [12], 16: [3, 4, 5], 17: [3, 4, 5], 18: [16, 17], 19: [14, 15],
                          20: [13, 19], 21: [20], 22: [21], 23: [22], 24: [20], 25: [24], 26: [6, 18, 23, 25], 27: [26], 28: [27], 29: [6, 18, 23, 25],
                          30: [6, 18, 23, 25], 31: [28, 29, 30], 32: [31], 33: [32]}

# 添加交叉关系约束
# 格式: {操作1: 操作2} 表示操作1和操作2不能并行执行
mutual_exclusion_constraints = {
    14: 15,  # 14和15是交叉关系，不能并行
    15: 14,  # 双向约束
    29: 30,  # 29和30是交叉关系，不能并行
    30: 29   # 双向约束
}

# 为pM24模型准备的数据
def get_model_data():
    """
    将原始数据转换为pM24模型所需的格式
    返回: n, jobs_operations, processing_times_adapted, machine_requirements_adapted, precedence_constraints_adapted, mutual_exclusion_adapted
    """
    # 假设只有一个作业，包含所有操作
    n = 1  # 作业数量
    
    # 操作数量
    n_operations = len(processing_times)
    jobs_operations = {1: n_operations}
    
    # 适配处理时间格式 (job_id, operation_id) -> time
    processing_times_adapted = {}
    for (job_id, op_id), time in processing_times.items():
        processing_times_adapted[job_id, op_id + 1] = time  # 操作从1开始编号
    
    # 适配机器需求格式
    machine_requirements_adapted = {}
    for (job_id, op_id), req in machine_requirements.items():
        machine_requirements_adapted[job_id, op_id + 1] = req
    
    # 适配前驱约束格式 - 转换为job内的约束格式
    precedence_constraints_adapted = {1: {}}
    for op_id, predecessors in precedence_constraints.items():
        if predecessors:  # 如果有前驱
            # 将操作ID从0开始转换为1开始
            precedence_constraints_adapted[1][op_id + 1] = [pred + 1 for pred in predecessors]
    
    # 适配交叉关系约束格式
    mutual_exclusion_adapted = {}
    for op1, op2 in mutual_exclusion_constraints.items():
        mutual_exclusion_adapted[op1 + 1] = op2 + 1  # 操作从1开始编号
    
    return n, jobs_operations, processing_times_adapted, machine_requirements_adapted, precedence_constraints_adapted, mutual_exclusion_adapted

# 为evaluate函数准备的原始数据（保持原格式）
def get_original_data():
    """
    返回原始格式的数据供evaluate函数使用
    """
    return processing_times, machine_requirements, precedence_constraints, mutual_exclusion_constraints
