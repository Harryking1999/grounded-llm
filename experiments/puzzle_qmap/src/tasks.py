"""Task construction without importing the learning stack."""


def make_environment(task, oracle_path=None):
    if task == "hanoi3":
        from experiments.hanoi_qmap.src.environment import Hanoi
        return Hanoi()
    if task == "cube2":
        from experiments.cube2_qmap.src.environment import Cube2
        return Cube2(oracle_path)
    raise ValueError(task)
