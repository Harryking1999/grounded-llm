"""Measure map branch scale where supervised ranking tokens are predicted."""

from argparse import ArgumentParser
from contextlib import nullcontext
import json
from pathlib import Path
import re
from statistics import median

import torch

from .blocks import FrozenBoardMap
from .blocks_data import demonstration_from_record as blocks_demonstration
from .data import demonstration_from_record as graph_demonstration, load_graph
from .graph import timeline_maps
from .text import chat_ids
from .train import build_reader, to_device


RANKING = re.compile(r"(?m)^Map-distance ranking to the goal, closest to farthest: (.+)\.$")


def ranking_prediction_positions(tokenizer, turn, template_kwargs):
    """Return full-sequence positions predicting the variable ranking tokens."""
    prefix = chat_ids(tokenizer, [{"role": "user", "content": turn.user_text}],
                      add_generation_prompt=True, **template_kwargs)
    complete = chat_ids(tokenizer, [{"role": "user", "content": turn.user_text},
                                    {"role": "assistant", "content": turn.answer_text}],
                        add_generation_prompt=False, **template_kwargs)
    content = tokenizer(turn.answer_text, add_special_tokens=False,
                        return_offsets_mapping=True)
    if complete[:len(prefix)] != prefix or complete[len(prefix):len(prefix) + len(content["input_ids"])] != content["input_ids"]:
        raise ValueError("Assistant ranking tokens do not align with chat template")
    match = RANKING.search(turn.answer_text)
    if match is None:
        raise ValueError("First turn has no ranking target")
    positions = [len(prefix) + index - 1
                 for index, (start, end) in enumerate(content["offset_mapping"])
                 if start < match.end(1) and end > match.start(1)]
    if not positions:
        raise ValueError("Ranking has no prediction positions")
    return complete, positions


def summarize_case(reader, timeline, token_ids, positions, device):
    """Capture un-gated and effective map residuals at ranking decisions."""
    per_layer = []
    hooks = []
    for layer in reader.conditioned_layers:
        record = {}
        attention = layer.map_attention

        def before(_module, inputs, *, item=record):
            item["hidden"] = inputs[0][:, positions].float()

        def branch(_module, _inputs, output, *, item=record):
            hidden_norm = torch.linalg.vector_norm(item["hidden"])
            item["ungated"] = float(torch.linalg.vector_norm(output[:, positions].float()) / hidden_norm)

        def after(_module, _inputs, output, *, item=record, adapter=attention):
            hidden = item.pop("hidden")
            item["effective"] = float(torch.linalg.vector_norm(output[:, positions].float() - hidden) /
                                      torch.linalg.vector_norm(hidden))
            item["gate"] = float(torch.tanh(adapter.gate))
            per_layer.append(item)

        hooks.extend((attention.register_forward_pre_hook(before),
                      attention.to_output.register_forward_hook(branch),
                      attention.register_forward_hook(after)))
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    try:
        with torch.inference_mode(), context:
            reader(timeline, input_ids=torch.tensor([token_ids], device=device),
                   attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long, device=device),
                   use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if len(per_layer) != len(reader.conditioned_layers):
        raise ValueError("Some map attention layers did not run")
    return per_layer


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--q-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-train-cases", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config != manifest["config"] or args.max_train_cases < 1:
        raise ValueError("Invalid map calibration contract")
    source = args.source_root if config["task"] == "graph" else args.q_checkpoint
    source_field = "source_root" if config["task"] == "graph" else "q_checkpoint"
    if source is None or source.resolve() != Path(manifest[source_field]).resolve():
        raise ValueError("Frozen map source differs from training manifest")
    saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    expected = {"config": config, "manifest": str(args.manifest.resolve()),
                "map_source": str(source.resolve()), "model_source": str(args.model_path.resolve())}
    if any(saved["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter was trained with another map contract")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    qmap = FrozenBoardMap.load(source) if config["task"] == "blocks" else None
    graphs = {}
    cases = []
    for record in manifest["records"]:
        if record["split"] != "train":
            continue
        if config["task"] == "blocks":
            demo = blocks_demonstration(qmap, record, config["maximum_demonstration_actions"])
        else:
            graph_id = record["graph_id"]
            if graph_id not in graphs:
                graphs[graph_id] = load_graph(source, graph_id)[:2]
            demo = graph_demonstration(*graphs[graph_id], record)
        turn = demo.turns[0]
        tokens, positions = ranking_prediction_positions(tokenizer, turn,
            config.get("chat_template_kwargs", {}))
        timeline = to_device(timeline_maps([turn.step], [0] * len(tokens)), device)
        layers = summarize_case(reader, timeline, tokens, positions, device)
        cases.append({"record": record, "ranking_tokens": len(positions),
                      "layers": layers, "median_ungated": median(x["ungated"] for x in layers),
                      "median_effective": median(x["effective"] for x in layers)})
        if len(cases) >= args.max_train_cases:
            break
    all_layers = [layer for case in cases for layer in case["layers"]]
    result = {"task": config["task"], "memory_mode": config["map"].get("memory_mode", "joint"),
              "value_scale": config["map"].get("value_scale", 1.0),
              "train_cases": len(cases),
              "median_ungated": median(x["ungated"] for x in all_layers),
              "median_effective": median(x["effective"] for x in all_layers),
              "cases": cases}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
