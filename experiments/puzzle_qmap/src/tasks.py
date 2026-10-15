"""Task construction without importing the learning stack."""


def make_environment(task, oracle_path=None):
    if task in {f"hanoi{disks}" for disks in range(1, 7)}:
        from experiments.hanoi_qmap.src.environment import Hanoi
        return Hanoi(int(task[5:]))
    if task in ("cube2", "cube2_qtm"):
        from experiments.cube2_qmap.src.environment import Cube2
        return Cube2(oracle_path, metric="QTM_URF_6" if task == "cube2_qtm" else "HTM_URF_9")
    raise ValueError(task)
