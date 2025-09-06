# MJSProblem.py
# 描述：定義多處理器作業車間排程問題的基本資料結構。

class Operation:
    """
    Attributes:
    - id: operation id
    - job_id: the job this operation belongs to
    - processing_times: dict {machine_id: processing_time}
    
   Property: 
    - machine_id: the machine this operation is assigned to (None if not assigned)
    - job_prev, job_next: pointers to previous and next operations in the same job
    - machine_prev, machine_next: pointers to previous and next operations in the same machine
    """
    def __init__(self, op_id, job_id, processing_times):
        self.id = op_id 
        self.job_id = job_id
        self.processing_times = processing_times
        self.machine_id = None 

        self.job_prev = None 
        self.job_next = None 
        self.machine_prev = None 
        self.machine_next = None 

    def __repr__(self):
        return f"Op({self.id}, Job:{self.job_id})"

    def get_processing_time(self):
        """Get the processing time on the assigned machine."""
        if self.machine_id is None:
            raise ValueError(f"Operation {self.id} has not been assigned to a machine.")
        if self.machine_id not in self.processing_times:
            raise ValueError(f"Machine {self.machine_id} is not a valid machine for operation {self.id}.")
        return self.processing_times[self.machine_id]

class Job:
    """
    Attributes:
    - id: job id
    - operations: list of Operation objects
    """
    def __init__(self, job_id, operations):
        self.id = job_id
        self.operations = operations
        for i in range(len(operations) - 1):
            operations[i].job_next = operations[i+1]
            operations[i+1].job_prev = operations[i]

    def __repr__(self):
        return f"Job({self.id})"
