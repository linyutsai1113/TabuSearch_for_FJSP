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

    jobs, num_machines = parse_jssp_instance('git-repos/jsp_framework/jsp/benchmark/instances/la04')
    print(f"Instance define completed: {len(jobs)} jobs, {num_machines} machines.")

    # generate an initial solution
    print("\n--- Generating a initial solution ---")
    initial_solution = generate_initial_solution(jobs, num_machines)
    print("\nInitial solution:")
    for m_id, ops in initial_solution.schedule.items():
        print(f"  Machine {m_id}: {ops}")

    # run Tabu Search
    solver = TabuSearchSolver(jobs, num_machines, max_iter_no_improve=1000)
    
    # measure time
    start_time = time.time()
    best_solution, best_makespan = solver.solve(initial_solution, max_iterations=1000)
    end_time = time.time()

    print(f"\nThe best Makespan: {best_makespan}")
    print("Final optimal solution:")
    for m_id, ops in best_solution.schedule.items():
        print(f"  Machine {m_id}: {ops}")
    
    print(f"\nTime: {end_time - start_time:.4f} s")

if __name__ == "__main__":
    main()
