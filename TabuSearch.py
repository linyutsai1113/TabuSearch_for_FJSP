# TabuSearch.py
# 描述：實現禁忌搜尋演算法的主體邏輯。

import random
import math
import copy

from MJSProblem import Job, Operation
from Solution import Solution
from GraphModel import DisjunctiveGraph
from Move import Move

class TabuSearchSolver:
    """
    Purpose: Implement the Tabu Search algorithm for the Multi-Processor Job Shop Scheduling Problem (MJSP).
    Attributes:
    - jobs: list of Job objects
    - num_machines: number of machines
    - graph: DisjunctiveGraph object
    - tabu_list: dict to store tabu moves with their expiration iteration
    - tabu_list_size: current size of the tabu list
    Methods:
    - solve: main method to perform the tabu search
    - find_best_neighbor_move: find the best non-tabu move in the neighborhood
    - estimate_move_makespan: estimate the makespan after a potential move using lower bounds
    """
    def __init__(self, jobs, num_machines, max_iter_no_improve=200):
        self.jobs = jobs
        self.num_machines = num_machines
        self.graph = DisjunctiveGraph(jobs, num_machines)
        self.max_iter_no_improve = max_iter_no_improve
        
        num_ops = len(self.graph.operations)
        num_jobs = len(jobs)

        # Based on s=25+ln(sum(|Mi|)/m+e-1)*(|O|/n) in the paper, but adjusted to fit smaller instances
        sum_flexibility = sum(len(op.processing_times) for op in self.graph.operations)
        
        # s=25+ln(sum(|Mi|)/m+e-1)*(|O|/n)
        if self.num_machines > 0 and num_jobs > 0:
            s = 25 + math.log(sum_flexibility / self.num_machines + math.e - 1) * (num_ops / num_jobs)
        else:
            s = 25 # Default value if no machines or jobs
        
        #print(f"Calculated tabu list base size: {s:.2f}")
        
        self.tabu_list_base_size = int(s)
        self.tabu_list_size = self.tabu_list_base_size
        self.tabu_list = {}

    def solve(self, initial_solution, max_iterations=2000):
        """
        Purpose: main method to perform the tabu search
        Input:
        - initial_solution: a feasible initial Solution object
        - max_iterations: maximum number of iterations
        Output:
        - best_solution: the best Solution found
        - best_makespan: the makespan of the best Solution found
        """
        print("\n--- Tabu Search Start ---")
        
        current_solution = initial_solution.copy()
        self.graph.build_graph_from_solution(current_solution)
        current_makespan = self.graph.calculate_makespan()
        
        best_solution = current_solution.copy()
        best_makespan = current_makespan
        
        print(f"\nInitial makespan: {best_makespan}")
        
        iter_since_best = 0
        
        for i in range(max_iterations):
            if iter_since_best >= self.max_iter_no_improve:
                print(f"\nCan't found better solution in  {self.max_iter_no_improve} iterations. Stopping search.")
                break

            # According to the paper's suggestion, dynamically adjust the tabu list size
            if i > 0 and self.tabu_list_base_size > 0 and i % int(1.3 * self.tabu_list_base_size) == 0:
                self.tabu_list_size = random.randint(
                    max(1, int(0.8 * self.tabu_list_base_size)),
                    int(1.2 * self.tabu_list_base_size)
                )

            best_move_info = self.find_best_neighbor_move(current_solution, i)
            
            if best_move_info is None:
                iter_since_best += 1
                continue
            
            move = best_move_info['move']
            
            # Save the current solution before applying the move
            solution_before_move = current_solution.copy()
            
            # Apply the move
            op_to_move = self.graph.op_map[move.op_id]
            old_machine = op_to_move.machine_id
            current_solution.schedule[old_machine].remove(move.op_id)
            current_solution.schedule[move.target_machine_id].insert(move.new_pos, move.op_id)
            
            # Update the graph and calculate the new Makespan
            self.graph.build_graph_from_solution(current_solution)
            current_makespan = self.graph.calculate_makespan()

            # check for cycles (makespan == inf indicates a cycle in this implementation) goback to previous solution if cycle detected
            if current_makespan == float('inf'):
                print(f"Iter. {i+1}: found cycles exist! Withdraw move: {move}")
                current_solution = solution_before_move
                self.graph.build_graph_from_solution(current_solution) 
                current_makespan = self.graph.calculate_makespan() 
                
                
                # add the move that caused the cycle to the tabu list
                pred_j_id = solution_before_move.schedule[move.target_machine_id][move.new_pos - 1] if move.new_pos > 0 else None
                self.tabu_list[(move.op_id, pred_j_id)] = i + self.tabu_list_size * 2 # for longer tabu tenure
                continue 

            # Update the tabu list (using the third type suggested in the paper)
            pred_j_id = current_solution.schedule[move.target_machine_id][move.new_pos - 1] if move.new_pos > 0 else None
            self.tabu_list[(move.op_id, pred_j_id)] = i + self.tabu_list_size
            
            # Remove expired tabu entries
            expired_keys = [k for k, v in self.tabu_list.items() if v < i]
            
            for k in expired_keys:
                del self.tabu_list[k]

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = current_solution.copy()
                iter_since_best = 0
                print(f"Iter. {i+1}: found better solution! Makespan: {best_makespan:.2f}")
            else:
                iter_since_best += 1
            
            if (i+1) % 50 == 0:
                print(f"Iter. {i+1}: Current Makespan: {current_makespan:.2f}, Best Makespan: {best_makespan:.2f}")

        print("\n--- Tabu Search End ---")
        return best_solution, best_makespan

    def find_best_neighbor_move(self, current_solution, current_iter):
        critical_ops = self.graph.critical_path
        
        if not critical_ops:
            return None

        
        moves_by_rank = {1: [], 2: [], 3: [], 4: [], 5: []}
        ops_to_consider = critical_ops 


        for op in self.graph.operations:
            current_machine_id = op.machine_id

            for target_machine_id in op.processing_times.keys(): # consider all capable machines
                machine_schedule = current_solution.schedule[target_machine_id]
                for pos in range(len(machine_schedule) + 1):
                    # avoid meaningless moves (to the same position or adjacent)
                    if target_machine_id == current_machine_id:
                        original_pos = machine_schedule.index(op.id)
                        if pos == original_pos or pos == (original_pos  + 1) or pos == (original_pos -1):
                            continue
                    
                    move = Move(op.id, target_machine_id, pos)
                    
                    pred_j_id = machine_schedule[pos - 1] if pos > 0 else None
                    
                    # check if the move is tabu
                    if self.tabu_list.get((move.op_id, pred_j_id), -1) > current_iter:
                        continue
                    
                    # According the Theorem 1 in paper, check the move feasibility
                    # ==========================================================
                    p_op = op.job_prev
                    f_op = op.job_next
                    j_op = self.graph.op_map.get(pred_j_id) if pred_j_id else self.graph.source
                    k_op = self.graph.op_map.get(machine_schedule[pos]) if pos < len(machine_schedule) else self.graph.sink
                    
                    # 檢查前置條件: j != f(i) and k != p(i)
                    if f_op and j_op.id == f_op.id: continue
                    if p_op and k_op.id == p_op.id: continue

                    is_feasible = True
                    # if (1): r_j < r_f(i) + p_f(i)
                    if f_op:
                        r_j = self.graph.release_times.get(j_op.id, 0)
                        r_f = self.graph.release_times.get(f_op.id, 0)
                        p_f = f_op.get_processing_time()
                        if not (r_j < r_f + p_f):
                            is_feasible = False
                    
                    # if (2): r_k + p_k > r_p(i)
                    if is_feasible and p_op:
                        r_k = self.graph.release_times.get(k_op.id, 0)
                        p_k = k_op.get_processing_time() if k_op.id != 'sink' else 0
                        r_p = self.graph.release_times.get(p_op.id, 0)
                        if not (r_k + p_k > r_p):
                            is_feasible = False
                    
                    if not is_feasible:
                        continue
                    # ==========================================================
                    
                    eval_results = self.estimate_move_makespan(op, target_machine_id, pos, current_solution)
                    theorem4_value = eval_results['theorem4_value']
                    final_lb = eval_results['final_lb']
                    
                    if op in self.graph.critical_path:
                        # According to the paper, classify moves into ranks R1-R5 (Section 6.2)
                        if theorem4_value < self.graph.makespan:
                            rank = 1
                        elif theorem4_value == self.graph.makespan:
                            rank = 2
                        else: # theorem4_value > self.graph.makespan
                            if final_lb < self.graph.makespan:
                                rank = 3
                            elif final_lb > self.graph.makespan:
                                rank = 4
                            else: # final_lb == self.graph.makespan
                                rank = 5
                    else:
                        rank = 5

                    moves_by_rank[rank].append({'move': move, 'lb': final_lb})


        # from the best rank, select the move with the lowest lower bound
        for rank in sorted(moves_by_rank.keys()):
            if moves_by_rank[rank]:
                return min(moves_by_rank[rank], key=lambda x: x['lb'])
        return None

    def estimate_move_makespan(self, op_to_move, target_machine_id, new_pos, solution):
        r, q = self.graph.release_times, self.graph.delivery_times
        p_op, f_op = op_to_move.job_prev, op_to_move.job_next
        
        schedule_on_target = solution.schedule[target_machine_id]
        j_op_id = schedule_on_target[new_pos - 1] if new_pos > 0 else 'source'
        k_op_id = schedule_on_target[new_pos] if new_pos < len(schedule_on_target) else 'sink'
        
        j_op = self.graph.op_map.get(j_op_id)
        k_op = self.graph.op_map.get(k_op_id)
        p_i_prime = op_to_move.processing_times[target_machine_id]
        
        # --- Theorem 4 (Section 6.3) ---
        r_p_term = (r.get(p_op.id, 0) + p_op.get_processing_time()) if p_op else 0
        r_j_term = (r.get(j_op.id, 0) + j_op.get_processing_time()) if j_op_id != 'source' else 0
        q_f_term = (q.get(f_op.id, 0) + f_op.get_processing_time()) if f_op else 0
        q_k_term = (q.get(k_op.id, 0) + k_op.get_processing_time()) if k_op_id != 'sink' else 0
        theorem4_value = max(r_p_term, r_j_term) + p_i_prime + max(q_f_term, q_k_term)
        # get op_to_move's predecessor and successor on its original machine (s, t)
        machine_ops = solution.schedule[op_to_move.machine_id]
        op_idx = machine_ops.index(op_to_move.id)
        s_op = self.graph.op_map[machine_ops[op_idx - 1]] if op_idx > 0 else self.graph.source
        t_op = self.graph.op_map[machine_ops[op_idx + 1]] if op_idx < len(machine_ops) - 1 else self.graph.sink
        
        # --- Lower Bound LB1 (Theorem 5, Section 6.3) ---
        r_hat_j = r.get(j_op.id, 0)
        # check if i in P(j)
        if j_op and self.graph.path_exists(op_to_move, j_op):
            r_t = r.get(t_op.id, 0)
            r_p_t = (r.get(t_op.job_prev.id, 0) + t_op.job_prev.get_processing_time()) if t_op.job_prev else 0
            r_s = (r.get(s_op.id, 0) + s_op.get_processing_time()) if s_op.id != 'source' else 0
            r_hat_j = r.get(j_op.id, 0) - r_t + max(r_p_t, r_s)
        
        q_hat_k = q.get(k_op.id, 0)
        # check if i in F(k)
        if k_op and self.graph.path_exists(k_op, op_to_move):
            q_s = q.get(s_op.id, 0)
            q_f_s = (q.get(s_op.job_next.id, 0) + s_op.job_next.get_processing_time()) if s_op.job_next else 0
            q_t = (q.get(t_op.id, 0) + t_op.get_processing_time()) if t_op.id != 'sink' else 0
            q_hat_k = q.get(k_op.id, 0) - q_s + max(q_f_s, q_t)
        
        r_hat_j_term = (r_hat_j + j_op.get_processing_time()) if j_op_id != 'source' else 0
        q_hat_k_term = (q_hat_k + k_op.get_processing_time()) if k_op_id != 'sink' else 0
        lb1 = max(r_p_term, r_hat_j_term) + p_i_prime + max(q_f_term, q_hat_k_term)
        
        # --- Lower Bound LB2 (Remark 1, Section 6.3) ---
        p_s = s_op.get_processing_time() if s_op.id != 'source' else 0
        p_t = t_op.get_processing_time() if t_op.id != 'sink' else 0
        lb2 = r.get(s_op.id, 0) + p_s + p_t + q.get(t_op.id, 0)
        
        return {'theorem4_value': theorem4_value, 'final_lb': max(lb1, lb2)}