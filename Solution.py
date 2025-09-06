# Solution.py
# 描述：定義排程解的結構，並提供生成初始解的方法。

import copy
from MJSProblem import Job, Operation

class Solution:
    """
    封裝一個排程解，即工序在各機器上的排列。
    """
    def __init__(self, schedule_dict):
        # schedule_dict: {machine_id: [op_id_1, op_id_2, ...]}
        self.schedule = schedule_dict
    
    def copy(self):
        return Solution(copy.deepcopy(self.schedule))
        
    def __repr__(self):
        return f"Solution({self.schedule})"

def generate_initial_solution(jobs, num_machines):
    """
    生成一個初始解，保留 FIFO 精神並加入啟發式規則。
    採用基於事件驅動的模擬，選擇下一個能最早開始的工序 (FCFS/FIFO)。
    如果有多個工序能同時開始，則選擇處理時間最短的 (SPT) 作為決勝規則。
    """
    operations = [op for job in jobs for op in job.operations]
    
    # step 1: machine assignment 
    operations.sort(key=lambda op: (len(op.processing_times), -sum(op.processing_times.values())/len(op.processing_times)))
    
    machine_workloads = {m: 0 for m in range(1, num_machines + 1)}
    
    for op in operations:
        best_machine = min(op.processing_times.keys(), key=lambda m: machine_workloads[m] + op.processing_times[m])
        op.machine_id = best_machine
        machine_workloads[best_machine] += op.get_processing_time()

    # step 2: scheduling with FIFO + SPT heuristic
    schedule_dict = {m: [] for m in range(1, num_machines + 1)}
    
    # tracking each machine's available time
    machine_avail_time = {m: 0 for m in range(1, num_machines + 1)}
    # tracking each operation's completion time
    op_completion_time = {}
    
    # tracking next operation index to schedule for each job
    next_op_idx = {job.id: 0 for job in jobs}
    
    num_scheduled_ops = 0
    while num_scheduled_ops < len(operations):
        
        candidates = []
        for job in jobs:
            # if the job still has unscheduled operations
            if next_op_idx[job.id] < len(job.operations):
                op = job.operations[next_op_idx[job.id]]
                
                # calculate ready time based on job precedence
                pred_op = op.job_prev
                ready_time = op_completion_time.get(pred_op.id, 0) if pred_op else 0
                
                # calculate earliest start time (est)
                est = max(ready_time, machine_avail_time.get(op.machine_id, 0))
                
                candidates.append({'op': op, 'est': est})

        # if no candidates found, break the loop
        if not candidates:
            break

        # heuristic selection rule:
        # 1. 主要排序：按最早開始時間 (est) 升序 (FIFO)
        # 2. 次要排序：按處理時間 (SPT) 升序
        best_candidate = min(candidates, key=lambda c: (c['est'], c['op'].get_processing_time()))
        
        op_to_schedule = best_candidate['op']
        start_time = best_candidate['est']
        
        machine_id = op_to_schedule.machine_id
        completion_time = start_time + op_to_schedule.get_processing_time()
        
        schedule_dict[machine_id].append(op_to_schedule.id)
        machine_avail_time[machine_id] = completion_time
        op_completion_time[op_to_schedule.id] = completion_time
        next_op_idx[op_to_schedule.job_id] += 1
        num_scheduled_ops += 1
            
    return Solution(schedule_dict)

