"""Aggregate recorded samples and export auditable dead-end case studies."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import blocks
from run import aggregate, evaluate, parse_output

ROOT = Path(__file__).resolve().parents[3]


def analyze(suite, runs):
    cases = {c["id"]: c for c in suite["cases"]}
    records, seen = [], set()
    run_info = []
    for filename in runs:
        payload = json.loads(filename.read_text())
        run_info.append({"path": str(filename.relative_to(ROOT)), "status": payload["status"],
                         "source_commit": payload["source_commit"], "calls": len(payload["cases"])})
        for record in payload["cases"]:
            key = record["case_id"], record["replicate"]
            if key in seen:
                raise ValueError(f"Duplicate instance/replicate {key}; select the intended run explicitly")
            seen.add(key)
            case = cases[record["case_id"]]
            if not record.get("api_error") and record.get("response_status") == "completed":
                # Recompute all verdicts, rather than trusting saved summary fields.
                verdict = evaluate(case, record["raw_output"])
                if verdict != record["verdict"]:
                    raise AssertionError(f"Verdict changed for {key}")
            records.append(record)
    summaries = aggregate(records, list(cases.values()))
    by_case = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    studies, path_failures = [], []
    for record in records:
        case = cases[record["case_id"]]
        verdict = record["verdict"]
        loss = verdict.get("first_irrecoverable_action")
        if loss and loss["kind"] == "untileable":
            solve = blocks.solver()
            before = blocks.from_grid(loss["before_grid"])
            after = blocks.from_grid(loss["after_grid"])
            minimum, reference_ids = solve(before)
            if minimum is None or solve(after)[0] is not None:
                raise AssertionError("Invalid dead-end diagnosis")
            # The reference from the same pre-action state demonstrates a concrete
            # alternative; it is never supplied to model inference.
            studies.append({"case_id": case["id"], "replicate": record["replicate"],
                            "condition": case["condition"], "initial_grid": case["grid"],
                            "budget": case["budget"], "first_loss": loss,
                            "minimum_before_loss": minimum,
                            "valid_alternative_suffix": [blocks.PLACEMENTS[i][1] for i in reference_ids],
                            "supported_prefix_length_capped": blocks.plausible_prefix(after, 3)
                            if blocks.locally_supported(after) else -1,
                            "case_successes": sum(r["verdict"].get("pass", False) for r in by_case[case["id"]]),
                            "case_calls": len(by_case[case["id"]])})
        elif case["condition"].startswith("path") and not verdict.get("pass"):
            path_failures.append({"case_id": case["id"], "replicate": record["replicate"],
                                  "failure_type": verdict.get("failure_type"),
                                  "shortest_moves": case["reference"]["length"],
                                  "attempted_moves": verdict.get("attempted_moves"),
                                  "illegal_move": verdict.get("illegal_move")})
    studies.sort(key=lambda item: (item["first_loss"]["locally_supported_after"],
                                  -item["case_successes"], item["supported_prefix_length_capped"]), reverse=True)
    independent_studies, included = [], set()
    for study in studies:
        if study["case_id"] not in included:
            included.add(study["case_id"])
            independent_studies.append(study)
    block_stats = {}
    for condition in ("blocks8", "blocks12", "blocks16"):
        selected = [c for c in cases.values() if c["condition"] == condition]
        if not selected:
            continue
        block_stats[condition] = {"boards": len(selected)}
        for key in ("occupied_cells", "minimum_actions", "initial_legal_actions", "dead_end_fraction",
                    "locally_supported_dead_end_fraction"):
            values = [c["diagnostics"][key] for c in selected]
            block_stats[condition][key] = {"min": min(values), "mean": statistics.mean(values), "max": max(values)}
    expected = {(c["id"], r) for c in cases.values() for r in range(1, c["replicates"] + 1)}
    return {"run_sources": run_info, "expected_calls": len(expected), "recorded_calls": len(records),
            "missing_samples": [{"case_id": c, "replicate": r} for c, r in sorted(expected - seen)],
            "summary": summaries, "selected_block_statistics": block_stats,
            "dead_end_sample_count": len(studies), "dead_end_independent_board_count": len(included),
            "case_studies": independent_studies, "path_failures": path_failures,
            "transport_errors": dict(Counter(r.get("api_error") for r in records if r.get("api_error"))),
            "incomplete_responses": sum(r.get("response_status") != "completed" and not r.get("api_error") for r in records)}


def render_study(study, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    before = blocks.from_grid(study["first_loss"]["before_grid"])
    action = study["first_loss"]["action"]
    after = blocks.from_grid(study["first_loss"]["after_grid"])
    alternate = study["valid_alternative_suffix"][0]
    figures = [(before, None, "Before the first losing move"),
               (before, action, "Model's legal removal"),
               (after, None, "After: no complete tiling exists"),
               (before, alternate, "Alternative with a valid completion")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.3))
    for ax, (mask, selected, title) in zip(axes, figures):
        highlighted = set()
        if selected:
            highlighted = {(selected["row"] + r, selected["col"] + c)
                           for r, c in blocks.SHAPES[selected["shape_id"]]}
        for r in range(10):
            for c in range(10):
                filled = bool(mask & (1 << (r * 10 + c)))
                color = "#e47742" if (r, c) in highlighted else "#25354a" if filled else "white"
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#d1d8df", linewidth=.6))
        ax.set(xlim=(0, 10), ylim=(10, 0), aspect="equal")
        ax.set_xticks([c + .5 for c in range(10)], range(10), fontsize=8)
        ax.set_yticks([r + .5 for r in range(10)], range(10), fontsize=8)
        ax.set_title(title, fontsize=10, pad=12)
        if selected:
            ax.set_xlabel(f"shape {selected['shape_id']} at (row {selected['row']}, col {selected['col']})", fontsize=9)
    fig.suptitle(f"{study['case_id']} | sample {study['replicate']} | losing step {study['first_loss']['step']}", fontsize=12)
    fig.tight_layout(rect=(0, .03, 1, .92))
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="experiments/gcml_counterexamples/runs/uncapped/suite.json")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", default="experiments/gcml_counterexamples/runs/uncapped/analysis.json")
    parser.add_argument("--render", type=int, default=0)
    args = parser.parse_args()
    suite = json.loads((ROOT / args.suite).read_text())
    report = analyze(suite, [ROOT / name for name in args.run])
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for study in report["case_studies"][:args.render]:
        render_study(study, output.parent / f"case_{study['case_id']}.png")
    print(json.dumps({"output": str(output), "recorded": report["recorded_calls"],
                      "missing": len(report["missing_samples"]),
                      "summary": {key: {field: value for field, value in summary.items() if field != 'per_case'}
                                  for key, summary in report["summary"].items()},
                      "dead_end_samples": report["dead_end_sample_count"],
                      "dead_end_boards": report["dead_end_independent_board_count"]}, indent=2))


if __name__ == "__main__":
    main()
