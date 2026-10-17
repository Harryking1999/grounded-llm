"""Build the root reading copy from the task-owned wording and small examples."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = (("blocks", "积木"), ("path", "寻路"), ("hanoi", "汉诺塔"), ("cube2", "二阶魔方"))


def render():
    parts = ["# 各任务文字、提示与监督答案总览\n\n"
             "本文件由 `python -m tasks.export_texts` 生成。原文维护在 `tasks/<task>/prompts/README.md`，"
             "示例维护在对应 `data/example.json`；积木逐轮文字示例由其准备脚本生成。"
             "项目进展见 [状态页](docs/PROJECT_STATUS_AND_TODO.md)。\n"]
    for name, title in TASKS:
        base = ROOT / "tasks" / name
        text = (base / "prompts/README.md").read_text(encoding="utf-8").strip()
        # Heading nesting only; the words themselves remain task-owned.
        text = "\n".join("##" + line if line.startswith("#") else line for line in text.splitlines())
        parts.append(f"\n## {title}\n\n来源：`tasks/{name}/prompts/README.md`\n\n{text}\n")
        sample = json.loads((base / "data/example.json").read_text(encoding="utf-8"))
        parts.append("\n### 数据说明例\n\n```json\n" + json.dumps(sample, ensure_ascii=False, indent=2) + "\n```\n")
        example = base / "prompts/example.md"
        if example.exists():
            parts.append("\n### 已渲染文字示例\n\n" + example.read_text(encoding="utf-8").strip() + "\n")
    return "\n".join(parts).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "TASK_TEXTS.md"
    content = render()
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            raise SystemExit("TASK_TEXTS.md is stale; run python -m tasks.export_texts")
        print("Task text reading copy is synchronized.")
    else:
        target.write_text(content, encoding="utf-8", newline="\n")
        print(target)


if __name__ == "__main__":
    main()
