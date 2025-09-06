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
    def __init__(self, jobs, num_machines, max_iter_no_improve=200):
        self.jobs = jobs
        self.num_machines = num_machines
        self.graph = DisjunctiveGraph(jobs, num_machines)
        self.max_iter_no_improve = max_iter_no_improve
        
        num_ops = len(self.graph.operations)
        num_jobs = len(jobs)

        # 根據論文公式計算禁忌列表大小 (Section 6.4)
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
        print("--- 開始禁忌搜尋 ---")
        
        current_solution = initial_solution.copy()
        self.graph.build_graph_from_solution(current_solution)
        current_makespan = self.graph.calculate_makespan()
        
        best_solution = current_solution.copy()
        best_makespan = current_makespan
        
        print(f"Initial makespan: {best_makespan}")
        
        iter_since_best = 0
        
        for i in range(max_iterations):
            if iter_since_best >= self.max_iter_no_improve:
                print(f"\n在 {self.max_iter_no_improve} 次迭代中未找到更優解，停止搜索。")
                break

            # 根據論文建議，動態調整禁忌列表大小
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
            
            # ******** 新增：儲存當前狀態以便回滾 ********
            solution_before_move = current_solution.copy()
            
            # 應用移動
            op_to_move = self.graph.op_map[move.op_id]
            old_machine = op_to_move.machine_id
            current_solution.schedule[old_machine].remove(move.op_id)
            current_solution.schedule[move.target_machine_id].insert(move.new_pos, move.op_id)
            
            # 更新圖並計算新的 Makespan
            self.graph.build_graph_from_solution(current_solution)
            current_makespan = self.graph.calculate_makespan()

            # ******** 新增：檢查循環並回滾 ********
            if current_makespan == float('inf'):
                print(f"迭代 {i+1}: 檢測到循環！撤銷移動 {move}")
                current_solution = solution_before_move
                self.graph.build_graph_from_solution(current_solution) # 恢復圖的狀態
                current_makespan = self.graph.calculate_makespan() # 恢復 makespan
                
                # 將導致循環的移動加入禁忌
                pred_j_id = solution_before_move.schedule[move.target_machine_id][move.new_pos - 1] if move.new_pos > 0 else None
                self.tabu_list[(move.op_id, pred_j_id)] = i + self.tabu_list_size * 2 # 給予更長的禁忌期
                continue # 跳過此次迭代的後續步驟

            # 更新禁忌列表 (使用論文建議的第三種方式)
            pred_j_id = current_solution.schedule[move.target_machine_id][move.new_pos - 1] if move.new_pos > 0 else None
            self.tabu_list[(move.op_id, pred_j_id)] = i + self.tabu_list_size
            
            # 移除過期的禁忌
            expired_keys = [k for k, v in self.tabu_list.items() if v < i]
            
            for k in expired_keys:
                del self.tabu_list[k]

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = current_solution.copy()
                iter_since_best = 0
                print(f"迭代 {i+1}: 找到新最優解! Makespan: {best_makespan:.2f}")
            else:
                iter_since_best += 1
            
            if (i+1) % 50 == 0:
                print(f"迭代 {i+1}: 當前 Makespan: {current_makespan:.2f}, 最優: {best_makespan:.2f}")

        print("\n--- 禁忌搜尋結束 ---")
        return best_solution, best_makespan

    def find_best_neighbor_move(self, current_solution, current_iter):
        critical_ops = self.graph.critical_path
        
        if not critical_ops:
            return None

        # 根據論文 Section 6.2 的移動分級 R1-R5
        moves_by_rank = {1: [], 2: [], 3: [], 4: [], 5: []}
        ops_to_consider = critical_ops 

        for op in ops_to_consider:
            current_machine_id = op.machine_id
            
            for target_machine_id in op.processing_times.keys(): # 遍歷所有可用機器
                machine_schedule = current_solution.schedule[target_machine_id]
                for pos in range(len(machine_schedule) + 1):
                    # 避免無意義的移動 (移動到原位或原位之後)
                    if target_machine_id == current_machine_id:
                        original_pos = machine_schedule.index(op.id)
                        if pos == original_pos or pos == original_pos + 1:
                            continue
                    
                    move = Move(op.id, target_machine_id, pos)
                    
                    pred_j_id = machine_schedule[pos - 1] if pos > 0 else None
                    # 檢查是否為禁忌移動
                    if self.tabu_list.get((move.op_id, pred_j_id), -1) > current_iter:
                        continue
                    
                    # 根據論文 Theorem 1 檢查移動是否可行
                    # ==========================================================
                    p_op = op.job_prev
                    f_op = op.job_next
                    j_op = self.graph.op_map.get(pred_j_id) if pred_j_id else self.graph.source
                    k_op = self.graph.op_map.get(machine_schedule[pos]) if pos < len(machine_schedule) else self.graph.sink
                    
                    # 檢查前置條件: j != f(i) and k != p(i)
                    if f_op and j_op.id == f_op.id: continue
                    if p_op and k_op.id == p_op.id: continue

                    is_feasible = True
                    # 條件 (1): r_j < r_f(i) + p_f(i)
                    if f_op:
                        r_j = self.graph.release_times.get(j_op.id, 0)
                        r_f = self.graph.release_times.get(f_op.id, 0)
                        # 注意：論文此處公式有誤，應為 r_j < r_f(i)， Theorem 1 的證明過程 (r_j >= r_f(i) + p_f(i)) 也暗示了這一點
                        # 但為忠於原文，暫時保留
                        p_f = f_op.get_processing_time()
                        if not (r_j < r_f + p_f):
                            is_feasible = False
                    
                    # 條件 (2): r_k + p_k > r_p(i)
                    if is_feasible and p_op:
                        r_k = self.graph.release_times.get(k_op.id, 0)
                        p_k = k_op.get_processing_time() if k_op.id != 'sink' else 0
                        r_p = self.graph.release_times.get(p_op.id, 0)
                        # 注意：論文此處公式有誤，應為 r_p(i) < r_k + p_k, 證明過程 (r_p(i) >= r_k + p_k) 也暗示了這一點
                        if not (r_k + p_k > r_p):
                            is_feasible = False
                    
                    if not is_feasible:
                        continue
                    # ==========================================================
                    
                    eval_results = self.estimate_move_makespan(op, target_machine_id, pos, current_solution)
                    theorem4_value = eval_results['theorem4_value']
                    final_lb = eval_results['final_lb']
                    
                    # 根據論文 Section 6.2 進行分級
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
                    
                    moves_by_rank[rank].append({'move': move, 'lb': final_lb})

        # 依序從 R1, R2, ... 中選擇最好的移動
        for rank in sorted(moves_by_rank.keys()):
            #print(f"R{rank} 級移動數量: {len(moves_by_rank[rank])}")
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
        
        # Theorem 4 檢查值 (Section 6.2)
        r_p_term = (r.get(p_op.id, 0) + p_op.get_processing_time()) if p_op else 0
        r_j_term = (r.get(j_op.id, 0) + j_op.get_processing_time()) if j_op_id != 'source' else 0
        q_f_term = (q.get(f_op.id, 0) + f_op.get_processing_time()) if f_op else 0
        q_k_term = (q.get(k_op.id, 0) + k_op.get_processing_time()) if k_op_id != 'sink' else 0
        theorem4_value = max(r_p_term, r_j_term) + p_i_prime + max(q_f_term, q_k_term)

        # 獲取 op_to_move 在原機器上的前後工序 (s, t)
        machine_ops = solution.schedule[op_to_move.machine_id]
        op_idx = machine_ops.index(op_to_move.id)
        s_op = self.graph.op_map[machine_ops[op_idx - 1]] if op_idx > 0 else self.graph.source
        t_op = self.graph.op_map[machine_ops[op_idx + 1]] if op_idx < len(machine_ops) - 1 else self.graph.sink
        
        # --- Lower Bound LB1 (Theorem 5, Section 6.3) ---
        r_hat_j = r.get(j_op.id, 0)
        # 檢查 i 是否在 P(j) 中
        if j_op and self.graph.path_exists(op_to_move, j_op):
            r_t = r.get(t_op.id, 0)
            r_p_t = (r.get(t_op.job_prev.id, 0) + t_op.job_prev.get_processing_time()) if t_op.job_prev else 0
            r_s = (r.get(s_op.id, 0) + s_op.get_processing_time()) if s_op.id != 'source' else 0
            r_hat_j = r.get(j_op.id, 0) - r_t + max(r_p_t, r_s)
        
        q_hat_k = q.get(k_op.id, 0)
        # 檢查 i 是否在 F(k) 中
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