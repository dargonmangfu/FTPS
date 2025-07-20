from pyscipopt import Model, quicksum, multidict

class DatasetModel:
    def pM24(n, m, jobs_operations, processing_times, machine_requirements, precedence_constraints):
        """
        Parameters:
            n: 作业数量
            m: 机器数量 (10台)
            jobs_operations: 字典，jobs_operations[i] = 作业i的操作数量
            processing_times: 字典，processing_times[i,j] = 作业i操作j的处理时间
            machine_requirements: 字典，machine_requirements[i,j] = 作业i操作j需要的机器数量
            precedence_constraints: 字典，precedence_constraints[i] = 作业i的前驱关系
        """

        model = Model("FJSP - Parallel Machines")
        
        # 变量定义
        s = {}  # s[i,j]: 作业i操作j的开始时间
        c = {}  # c[i,j]: 作业i操作j的完成时间  
        job_c = {}  # job_c[i]: 作业i的完成时间
        x = {}  # x[i,j,t]: 作业i操作j是否在时间t开始 (离散时间)
        y = {}  # y[k,t]: 机器k在时间t是否被占用
        
        # 时间范围 (根据问题规模调整)
        T = sum(processing_times.values()) + 100  # 估算总时间上界
        
        # 变量创建
        for i in range(1, n+1):
            job_c[i] = model.addVar(lb=0, vtype="C", name=f"job_c_{i}")
            for j in range(1, jobs_operations[i]+1):
                s[i,j] = model.addVar(lb=0, vtype="C", name=f"s_{i}_{j}")
                c[i,j] = model.addVar(lb=0, vtype="C", name=f"c_{i}_{j}")
                
                # 离散时间变量 (如果使用时间索引方法)
                for t in range(T):
                    x[i,j,t] = model.addVar(vtype="B", name=f"x_{i}_{j}_{t}")
        
        # 机器占用变量
        for k in range(1, m+1):
            for t in range(T):
                y[k,t] = model.addVar(vtype="B", name=f"y_{k}_{t}")

        # 约束条件
        # 1. 每个操作只能在一个时间点开始
        for i in range(1, n+1):
            for j in range(1, jobs_operations[i]+1):
                model.addCons(quicksum(x[i,j,t] for t in range(T)) == 1, f"start_once_{i}_{j}")

        # 2. 操作时间关系
        for i in range(1, n+1):
            for j in range(1, jobs_operations[i]+1):
                # 开始时间
                model.addCons(s[i,j] == quicksum(t * x[i,j,t] for t in range(T)), f"start_time_{i}_{j}")
                # 完成时间
                model.addCons(c[i,j] == s[i,j] + processing_times[i,j], f"completion_time_{i}_{j}")

        # 3. 机器需求约束 - 每个操作需要指定数量的机器
        for i in range(1, n+1):
            for j in range(1, jobs_operations[i]+1):
                required_machines = machine_requirements[i,j]
                for t in range(T):
                    if t < T - processing_times[i,j]:  # 确保操作能在时间范围内完成
                        # 如果操作在时间t开始，则需要占用required_machines台机器
                        model.addCons(
                            quicksum(y[k,tau] for k in range(1, m+1) 
                                    for tau in range(t, t + processing_times[i,j])) >= 
                            required_machines * processing_times[i,j] * x[i,j,t],
                            f"machine_requirement_{i}_{j}_{t}"
                        )

        # 4. 机器容量约束 - 同一时间每台机器最多被一个操作占用
        for k in range(1, m+1):
            for t in range(T):
                model.addCons(y[k,t] <= 1, f"machine_capacity_{k}_{t}")

        # 5. 作业内操作顺序约束 - 替换原来的简单线性约束
        for i in range(1, n+1):
            if i in precedence_constraints:
                for j in range(1, jobs_operations[i]+1):
                    if j in precedence_constraints[i]:
                        predecessors = precedence_constraints[i][j]
                        # 对于每个前驱操作，当前操作必须在所有前驱完成后才能开始
                        for pred in predecessors:
                            model.addCons(s[i,j] >= c[i,pred], f"precedence_{i}_{j}_after_{pred}")
            else:
                # 如果没有指定前驱关系，使用默认的线性顺序
                for j in range(2, jobs_operations[i]+1):
                    model.addCons(s[i,j] >= c[i,j-1], f"precedence_{i}_{j}")
        
        # 6. 作业完成时间
        for i in range(1, n+1):
            last_operation = jobs_operations[i]
            model.addCons(job_c[i] >= c[i,last_operation], f"job_completion_{i}")

        # 目标函数：最小化最大完成时间
        model.setObjective(quicksum(job_c[i] for i in range(1, n+1)), "minimize")

        return model
    from pyscipopt import Model, quicksum, multidict

    def pM2(n, m, jobs_operations, processing_times, machine_sets, precedence_constraints):
        """
        Parameters:
            n: 作业数量
            m: 机器数量
            jobs_operations: 字典，jobs_operations[i] = 作业i的操作数量
            processing_times: 字典，processing_times[i,j,k] = 作业i操作j在机器k上的处理时间
            machine_sets: 字典，machine_sets[i,j] = 作业i操作j可用的机器集合
            precedence_constraints: 字典，precedence_constraints[i] = {操作: [前驱操作列表]}
        """

        model = Model("FJSP - Flexible Job Shop")
        
        # 变量定义
        s = {}      # s[i,j,k]: 作业i操作j在机器k上的开始时间
        c = {}      # c[i,j,k]: 作业i操作j在机器k上的完成时间  
        v = {}      # v[i,j,k]: 作业i操作j是否在机器k上执行 (二进制变量)
        job_c = {}  # job_c[i]: 作业i的完成时间
        z = {}      # z[i,j,h,g,k]: 操作优先级变量
        
        BigM = sum(max(processing_times[i,j,k] for k in machine_sets[i,j]) 
                for i in range(1, n+1) for j in range(1, jobs_operations[i]+1)) + 1000
        
        # 变量创建
        for i in range(1, n+1):
            job_c[i] = model.addVar(lb=0, vtype="C", name=f"job_c_{i}")
            for j in range(1, jobs_operations[i]+1):
                for k in machine_sets[i,j]:
                    s[i,j,k] = model.addVar(lb=0, vtype="C", name=f"s_{i}_{j}_{k}")
                    c[i,j,k] = model.addVar(lb=0, vtype="C", name=f"c_{i}_{j}_{k}")
                    v[i,j,k] = model.addVar(vtype="B", name=f"v_{i}_{j}_{k}")

        # 机器冲突变量
        for k in range(1, m+1):
            operations_on_k = [(i,j) for i in range(1, n+1) 
                            for j in range(1, jobs_operations[i]+1) 
                            if k in machine_sets[i,j]]
            for idx1, (i,j) in enumerate(operations_on_k):
                for idx2, (h,g) in enumerate(operations_on_k):
                    if idx1 != idx2:
                        z[i,j,h,g,k] = model.addVar(vtype="B", name=f"z_{i}_{j}_{h}_{g}_{k}")

        # 约束条件
        # 1. 每个操作必须且只能在一台机器上执行
        for i in range(1, n+1):
            for j in range(1, jobs_operations[i]+1):
                model.addCons(
                    quicksum(v[i,j,k] for k in machine_sets[i,j]) == 1, 
                    f"one_machine_{i}_{j}"
                )

        # 2. 操作时间关系 - 只有被选中的机器上的时间变量才有效
        for i in range(1, n+1):
            for j in range(1, jobs_operations[i]+1):
                for k in machine_sets[i,j]:
                    # 如果操作不在机器k上执行，则开始和完成时间为0
                    model.addCons(
                        s[i,j,k] <= BigM * v[i,j,k], 
                        f"start_if_selected_{i}_{j}_{k}"
                    )
                    model.addCons(
                        c[i,j,k] <= BigM * v[i,j,k], 
                        f"complete_if_selected_{i}_{j}_{k}"
                    )
                    # 如果操作在机器k上执行，则完成时间 = 开始时间 + 处理时间
                    model.addCons(
                        c[i,j,k] >= s[i,j,k] + processing_times[i,j,k] - BigM * (1 - v[i,j,k]),
                        f"processing_time_{i}_{j}_{k}"
                    )

        # 3. 机器容量约束 - 同一台机器上的操作不能重叠
        for k in range(1, m+1):
            operations_on_k = [(i,j) for i in range(1, n+1) 
                            for j in range(1, jobs_operations[i]+1) 
                            if k in machine_sets[i,j]]
            for idx1, (i,j) in enumerate(operations_on_k):
                for idx2, (h,g) in enumerate(operations_on_k):
                    if idx1 != idx2:
                        # 操作(i,j)在操作(h,g)之前 或 操作(h,g)在操作(i,j)之前
                        model.addCons(
                            c[i,j,k] <= s[h,g,k] + BigM * (2 - v[i,j,k] - v[h,g,k] + z[i,j,h,g,k]),
                            f"no_overlap_1_{i}_{j}_{h}_{g}_{k}"
                        )
                        model.addCons(
                            c[h,g,k] <= s[i,j,k] + BigM * (3 - v[i,j,k] - v[h,g,k] - z[i,j,h,g,k]),
                            f"no_overlap_2_{i}_{j}_{h}_{g}_{k}"
                        )

        # 4. 作业内操作顺序约束
        for i in range(1, n+1):
            if i in precedence_constraints:
                for j in range(1, jobs_operations[i]+1):
                    if j in precedence_constraints[i]:
                        predecessors = precedence_constraints[i][j]
                        for pred in predecessors:
                            # 当前操作j必须在所有前驱操作完成后开始
                            for k_curr in machine_sets[i,j]:
                                for k_pred in machine_sets[i,pred]:
                                    model.addCons(
                                        s[i,j,k_curr] >= c[i,pred,k_pred] - BigM * (2 - v[i,j,k_curr] - v[i,pred,k_pred]),
                                        f"precedence_{i}_{j}_after_{pred}_{k_curr}_{k_pred}"
                                    )
            else:
                # 默认线性顺序
                for j in range(2, jobs_operations[i]+1):
                    for k_curr in machine_sets[i,j]:
                        for k_prev in machine_sets[i,j-1]:
                            model.addCons(
                                s[i,j,k_curr] >= c[i,j-1,k_prev] - BigM * (2 - v[i,j,k_curr] - v[i,j-1,k_prev]),
                                f"sequence_{i}_{j}_{k_curr}_{k_prev}"
                            )

        # 5. 作业完成时间
        for i in range(1, n+1):
            last_operation = jobs_operations[i]
            for k in machine_sets[i,last_operation]:
                model.addCons(
                    job_c[i] >= c[i,last_operation,k],
                    f"job_completion_{i}_{k}"
                )

        # 目标函数：最小化最大完成时间 (makespan)
        makespan = model.addVar(lb=0, vtype="C", name="makespan")
        for i in range(1, n+1):
            model.addCons(makespan >= job_c[i], f"makespan_constraint_{i}")
        
        model.setObjective(makespan, "minimize")

        return model





