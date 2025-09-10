# main.py
# 描述：主執行檔，用於設置問題實例並啟動禁忌搜尋演算法。

import time
from MJSProblem import Job, Operation
from Solution import generate_initial_solution
from TabuSearch import TabuSearchSolver

def parse_jssp_instance(filepath):
    """
    Purpose: Parse a JSSP instance file.
    Input: filepath - path to the instance file
    Output: 
    - jobs - list of Job objects
    - num_machines - number of machines
    Note: Example for JSSP instance
    
    #+++++++++++++++++++++++++++++
    # instance la01
    #+++++++++++++++++++++++++++++
    # Lawrence 10x5 instance (Table 3, instance 1); also called (setf1) or (F1)
    10 5
    1 21 0 53 4 95 3 55 2 34
    0 21 3 52 4 16 2 26 1 71
    3 39 4 98 1 42 2 31 0 12
    1 77 0 55 4 79 2 66 3 77
    0 83 3 34 2 64 1 19 4 37
    1 54 2 43 4 79 0 92 3 62
    3 69 4 77 1 87 2 87 0 93
    2 38 0 60 1 41 3 24 4 83
    3 17 1 49 4 25 0 44 2 98
    4 77 3 79 2 43 1 75 0 96

    """
    with open(filepath, 'r') as f:
        lines = f.read().strip().splitlines()

    num_jobs, num_machines = map(int, lines[4].split())

    jobs = []

    # parse operations for each job
    for job_index, line in enumerate(lines[5:5 + num_jobs]):
        tokens = list(map(int, line.strip().split()))
        operations = []
        for op_index in range(0, len(tokens), 2):
            machine_id = tokens[op_index] + 1  # Convert to 1-indexed
            processing_time = tokens[op_index + 1]
            op_id = f"{job_index + 1}-{op_index // 2 + 1}"

            op = Operation(op_id, job_index + 1, {machine_id: processing_time})
            operations.append(op)

        job = Job(job_index + 1, operations)
        jobs.append(job)

    return jobs, num_machines


def parse_fjsp_instance(filepath):
    """
    Purpose: parse a FJSP instance file.
    Input: filepath - path to the instance file
    Output: 
    - jobs - list of Job objects
    - num_machines - number of machines
    Note: Example for FJSP instance
    - first row: [job number] [machine number] (optional: average number of machines per operation)
    - follow row represent each job:
      - first number (n_ops): number of operations in this job
      - then according to k, there are k pairs of numbers (machine,processing time) that specify which are the machines and the processing times 

    Example: Fisher and Thompson 6x6 instance, alternate name (mt06)

    ```
    6   6   1   
    6   1   3   1   1   1   3   1   2   6   1   4   7   1   6   3   1   5   6   
    6   1   2   8   1   3   5   1   5   10  1   6   10  1   1   10  1   4   4   
    6   1   3   5   1   4   4   1   6   8   1   1   9   1   2   1   1   5   7   
    6   1   2   5   1   1   5   1   3   5   1   4   3   1   5   8   1   6   9   
    6   1   3   9   1   2   3   1   5   5   1   6   4   1   1   3   1   4   1   
    6   1   2   3   1   4   3   1   6   9   1   1   10  1   5   4   1   3   1   
    ```

    first row: 6 jobs, 6 machines, and 1 machine per operation  
    second row: job 1 has 6 operations; the first operation can be processed by 1 machine, that is machine 3 with processing time 1.

    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    first_line_tokens = lines[0].split()
    num_jobs = int(first_line_tokens[0])
    num_machines = int(first_line_tokens[1])

    jobs = []
    for job_index, line in enumerate(lines[1:1 + num_jobs]):
        tokens = list(map(int, line.split()))
        
        operations = []
        num_ops = tokens[0]
        
        token_ptr = 1 
        for op_index in range(num_ops):
            op_id = f"{job_index + 1}-{op_index + 1}"
            
            num_machine_options = tokens[token_ptr]
            token_ptr += 1
            
            processing_times = {}
            for _ in range(num_machine_options):
                machine_id = tokens[token_ptr]
                processing_time = tokens[token_ptr + 1]
                processing_times[machine_id] = processing_time
                token_ptr += 2
            
            op = Operation(op_id, job_index + 1, processing_times)
            operations.append(op)
        
        job = Job(job_index + 1, operations)
        jobs.append(job)

    return jobs, num_machines


def main():
    """
    Purpose: main function to set up the problem instance and run the Tabu Search algorithm.
    Logic:
    1. Prase a JSSP instance file to get jobs and number of machines. (You need to change to your own file path)
    2. Generate an initial solution using a heuristic method.
    3. Create a Tabu Search solver instance and run the algorithm.
    4. Print the results.
    """
    print("--- Setting a Problem instance ---")

    # fjsp instance
    filepath = 'git-repos/TabuSearch_for_FJSP/benchmark/fjs_data/Hurink/Hurink_edata_la02.fjs' 
    jobs, num_machines = parse_fjsp_instance(filepath)
    
    # jssp instance
    # filepath = 'benchmark/instances/la02'
    # jobs, num_machines = parse_jssp_instance(filepath)

    print(f"Instance '{filepath}' define completed: {len(jobs)} jobs, {num_machines} machines.")

    print("\n--- Generating a initial solution ---")
    initial_solution = generate_initial_solution(jobs, num_machines)
    print("\nInitial solution:")
    for m_id, ops in initial_solution.schedule.items():
        print(f"  Machine {m_id}: {ops}")

    # run Tabu Search
    solver = TabuSearchSolver(jobs, num_machines, max_iter_no_improve=2000)
    
    # measure time
    start_time = time.time()
    best_solution, best_makespan = solver.solve(initial_solution, max_iterations=20000)
    end_time = time.time()

    print(f"\nThe best Makespan: {best_makespan}")
    print("Final optimal solution:")
    for m_id, ops in best_solution.schedule.items():
        print(f"  Machine {m_id}: {ops}")
    
    print(f"\nTime: {end_time - start_time:.4f} s")

if __name__ == "__main__":
    main()
