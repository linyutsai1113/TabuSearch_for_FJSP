# GraphModel.py
# 描述：實現析取圖模型，採用模組化設計，將通用圖演算法與排程邏輯分離。

from collections import deque, defaultdict
from MJSProblem import Operation

class _DirectedGraph:
    """
    Purpose: General directed graph implementation.
    This class provides basic functionalities for directed graphs, including:
    - Adding nodes and edges
    - Topological sorting
    - Calculating longest paths from a given start node
    Note: This class is designed to be used as a utility within the DisjunctiveGraph class.
    """

    def __init__(self) -> None:
        self.__adjacency = defaultdict(list)
        self.__nodes = set()

    def add_node(self, node):
        self.__nodes.add(node)

    def add_edge(self, node_from, node_to): # directed edge
        self.__adjacency[node_from].append(node_to)
        self.add_node(node_from)
        self.add_node(node_to)

    def topological_sort(self) -> list:
        in_degrees = {node: 0 for node in self.__nodes}
        for node_list in self.__adjacency.values():
            for adj_node in node_list:
                in_degrees[adj_node] += 1
        
        queue = deque([node for node, degree in in_degrees.items() if degree == 0])
        res = []
        
        while queue:
            node = queue.popleft()
            res.append(node)
            for adj_node in self.__adjacency.get(node, []):
                in_degrees[adj_node] -= 1
                if in_degrees[adj_node] == 0:
                    queue.append(adj_node)

        # if the number of nodes in the topological order is not equal to total nodes, there is a cycle
        if len(res) != len(self.__nodes):
            return None # graph has cycles
        return res

    def calculate_all_longest_paths(self, start_node, weight_func):
        """
        Purpose: calculate the longest paths from start_node to all other nodes.
        Inputs:
        - start_node: the node to start from
        - weight_func: a function that takes a node and returns its weight (processing time)
        Returns: dict {node: longest_path_length} or None if graph has cycles?
        Logic:
        1. do a topological sort of the graph.
        2. use dynamic programming to calculate longest paths.
        3. return the longest path lengths. or None if graph has cycles.
        """
        sorted_nodes = self.topological_sort()
        if not sorted_nodes:
            return None # graph has cycles

        dist = {node: float('-inf') for node in self.__nodes}
        if start_node in dist:
            dist[start_node] = 0.0

        for node in sorted_nodes:
            if dist.get(node, float('-inf')) > float('-inf'):
                # weight is the weight of edges starting from node, i.e., the processing time of node
                weight = weight_func(node)
                for adj_node in self.__adjacency.get(node, []):
                    new_length = dist[node] + weight
                    if dist[adj_node] < new_length:
                        dist[adj_node] = new_length
        return dist


class DisjunctiveGraph:
    """
    Purpose: (Extended) Disjunctive graph model for the Multi-Processor Job Shop Scheduling Problem (MJSP).
    Inputs:
    - jobs: list of Job objects
    - num_machines: total number of machines
    Attributes:
    - operations: list of all Operation objects
    - op_map: mapping from operation id to Operation object
    - source, sink: special nodes representing the start and end of the schedule
    - release_times: dict {op_id: release_time} NOTE: used in calculating forward graph makespan
    - delivery_times: dict {op_id: delivery_time} NOTE: used in calculating reverse graph makespan
    - makespan: float
    - critical_path: list of Operation objects on the critical path
    Methods:
    - build_graph_from_solution(solution): build or update machine links based on a given solution
    - calculate_makespan(): calculate the makespan using the _DirectedGraph class
    - path_exists(start_op, end_op): check if a path exists from start_op to end_op
    """
    def __init__(self, jobs, num_machines):
        self.jobs = jobs
        self.num_machines = num_machines
        self.operations = [op for job in jobs for op in job.operations]
        self.op_map = {op.id: op for op in self.operations}
        self.source = Operation('source', -1, {'source': 0}); self.source.machine_id = 'source'
        self.sink = Operation('sink', -1, {'sink': 0}); self.sink.machine_id = 'sink'
        self.op_map.update({'source': self.source, 'sink': self.sink})
        
        self.all_nodes = self.operations + [self.source, self.sink]
        
        self.release_times = {} 
        self.delivery_times = {}
        self.makespan = 0
        self.critical_path = []

    def build_graph_from_solution(self, solution):
        """
        Purpose: build or update the disjunctive graph based on a given solution.
        Inputs:
        - solution: Solution object
        Logic: 
        1. reset all machine links of operations.
        2. for each machine, link the operations in the order specified by the solution's schedule.
        - solution: Solution object
        Note: we assume that all operation IDs in the solution's schedule dictionary exist in op_map. 
        """
        for op in self.operations:
            op.machine_prev = op.machine_next = None 
        for machine_id, op_ids in solution.schedule.items():
            for op_id in op_ids:
                if op_id in self.op_map:
                    self.op_map[op_id].machine_id = machine_id
        for machine_id, op_ids in solution.schedule.items():
            op_seq = [self.op_map[op_id] for op_id in op_ids if op_id in self.op_map]
            for i in range(len(op_seq) - 1):
                op_seq[i].machine_next = op_seq[i+1]
                op_seq[i+1].machine_prev = op_seq[i]

    def calculate_makespan(self):
        """
        Purpose: culculate the makespan of the current graph.
        Returns: makespan (float)
        Logic:
        1. initialize forward and reverse graphs
        2. use forward graph to calculate all release times (r_i)
        3. use reverse graph to calculate all delivery times (q_i)
        4. search for critical path.
        5. the length of the longest path from source to sink is the makespan. (also the length of critical path)
        Note: if the graph has cycles, makespan is set to infinity.
        """
        forward_graph = _DirectedGraph()
        reverse_graph = _DirectedGraph()
        all_nodes = self.all_nodes

        for node in all_nodes:
            forward_graph.add_node(node)
            reverse_graph.add_node(node)

        # add edges according to job and machine precedence
        for op in self.operations:
            # jobs constraints
            job_pred = op.job_prev if op.job_prev else self.source
            forward_graph.add_edge(job_pred, op)
            reverse_graph.add_edge(op, job_pred)
            
            # machine constraints
            machine_pred = op.machine_prev if op.machine_prev else self.source
            forward_graph.add_edge(machine_pred, op)
            reverse_graph.add_edge(op, machine_pred)

            # connect to sink if no successors
            if not op.job_next:
                forward_graph.add_edge(op, self.sink)
                reverse_graph.add_edge(self.sink, op)
            if not op.machine_next:
                forward_graph.add_edge(op, self.sink)
                reverse_graph.add_edge(self.sink, op)

        # NOTE: below is calculating release times (r_i) using forward graph

        weight_func = lambda op: op.get_processing_time() 
        r_times_raw = forward_graph.calculate_all_longest_paths(self.source, weight_func)

        if r_times_raw is None: # graph has cycles
            self.makespan = float('inf')
            return self.makespan
        
        self.release_times = {op.id: time for op, time in r_times_raw.items()}
        self.makespan = self.release_times.get(self.sink.id, 0)
        
        # NOTE: below is calculating delivery times (q_i) using reverse graph
        # Because the weight in reverse graph should be the weight of the edge's end node, 
        # which means the processing time of that node.
        # reverse graph is from sink to source, edge (u,v) weight is the processing time of v
        # so when we calculate, we need to use the op_map to get the processing time of the end node

        rev_weight_func = lambda op: self.op_map[op.id].get_processing_time()
        q_times_raw = reverse_graph.calculate_all_longest_paths(self.sink, rev_weight_func)
        
        if q_times_raw is None: # graph has cycles
            self.makespan = float('inf')
            return self.makespan

        self.delivery_times = {op.id: time for op, time in q_times_raw.items()}

        # search for critical path
        # critical path: all nodes i where r_i + p_i + q_i = Cmax
        self.critical_path = [op for op in self.operations if abs(self.release_times.get(op.id, 0) + op.get_processing_time() + self.delivery_times.get(op.id, 0) - self.makespan) < 1e-6]
        
        return self.makespan

    def path_exists(self, start_op, end_op):
        """
        Purpose: check if there is a path from start_op to end_op. 
        Inputs:
        - start_op, end_op: Operation objects
        Returns: True/False
        Logic: use BFS to check if a path exists. 
        """
        if not start_op or not end_op: return False
        if start_op.id == end_op.id: return True
        
        queue = deque([start_op])
        visited = {start_op.id}
        
        while queue:
            current_op = queue.popleft()
            successors = []
            if current_op.job_next: successors.append(current_op.job_next)
            if current_op.machine_next: successors.append(current_op.machine_next)
                
            for succ in successors:
                if succ.id == end_op.id: return True
                if succ.id not in visited:
                    visited.add(succ.id)
                    queue.append(succ)

        return False