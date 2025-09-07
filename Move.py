# Move.py
# 描述：定義一個「移動」操作的資料結構。

class Move:
    """
    Represents a move operation in the Tabu Search algorithm.
    Attributes:
    - op_id: ID of the operation being moved
    - target_machine_id: ID of the target machine
    """
    def __init__(self, op_id, target_machine_id, new_pos):
        self.op_id = op_id
        self.target_machine_id = target_machine_id
        self.new_pos = new_pos

    def __repr__(self):
        return f"Move(Op:{self.op_id} -> M:{self.target_machine_id}, Pos:{self.new_pos})"
