# Solution.py
# 描述：定義排程解的結構，並提供生成初始解的方法。

import copy
from MJSProblem import Job, Operation

class Solution:
    """
    Packaging a schedule solution, i.e., the arrangement of operations on machines.
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
    Purpose: gnerate an initial solution using a heuristic method. (FIFO + SPT)
    Input: 
    - jobs: list of Job objects
    - num_machines: number of machines
    Output: Solution object
    Logic:
    1. Machine assignment: assign each operation to a machine to balance workloads.
    2. Scheduling: use earliest start time (FIFO) as the primary rule, and shortest processing time (SPT) as the tie-breaker.
    Note: This is a simple heuristic and can be improved. Also, the initial solution quality may affect the Tabu Search performance. should be a feasible solution.
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
        # 1. main sort: by earliest start time (est) ascending (FIFO)
        # 2. secondary sort: by shortest processing time (SPT) ascending
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

