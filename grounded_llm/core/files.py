"""Atomic small JSON outputs shared by ML and the independent transfer runtime."""

import json
from pathlib import Path


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def atomic_save(payload, path):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
