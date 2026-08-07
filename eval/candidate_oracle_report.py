import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def method_from_path(path: str) -> str:
    name = os.path.basename(path)
    for suffix in ("_gt.jsonl", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def candidate_valid(candidate: Dict[str, Any], clearance_threshold: float) -> bool:
    margin = finite_float(candidate.get("reaction_margin_gt"))
    clearance = finite_float(candidate.get("min_clearance_gt"))
    if margin is None or clearance is None:
        return False
    return clearance >= clearance_threshold


def summarize(rows: List[Dict[str, Any]], clearance_threshold: float, max_examples: int):
    total_rows = len(rows)
    rows_with_candidates = 0
    valid_frames = 0
    selected_rmvr_frames = 0
    safe_available_frames = 0
    safe_missed_frames = 0
    oracle_selected_frames = 0
    collision_candidate_frames = 0
    gaps = []
    examples = []

    for row in rows:
        candidates = row.get("candidates") or []
        if not candidates:
            continue
        rows_with_candidates += 1
        selected_id = int(row.get("selected_id", -1))
        selected = next((c for c in candidates if int(c.get("id", -2)) == selected_id), None)
        valid = [c for c in candidates if candidate_valid(c, clearance_threshold)]
        if not valid or selected is None:
            continue
        valid_frames += 1
        oracle = max(valid, key=lambda c: finite_float(c.get("reaction_margin_gt")) or -1e9)
        selected_margin = finite_float(selected.get("reaction_margin_gt"))
        oracle_margin = finite_float(oracle.get("reaction_margin_gt"))
        if selected_margin is None or oracle_margin is None:
            continue
        gap = oracle_margin - selected_margin
        gaps.append(gap)
        selected_rmvr = selected_margin < 0.0
        safe_available = oracle_margin > 0.0
        safe_missed = selected_rmvr and safe_available
        selected_rmvr_frames += int(selected_rmvr)
        safe_available_frames += int(safe_available)
        safe_missed_frames += int(safe_missed)
        oracle_selected_frames += int(int(oracle.get("id", -3)) == selected_id)
        collision_candidate_frames += int(bool(selected.get("collision_gt")))
        if safe_missed or gap > 0.5:
            examples.append(
                {
                    "depth_count": row.get("depth_count"),
                    "time": row.get("time"),
                    "goal_distance": row.get("goal_distance"),
                    "selected_id": selected_id,
                    "selected_type": selected.get("type"),
                    "selected_margin_gt": selected_margin,
                    "selected_margin_pred": finite_float(selected.get("margin_pred")),
                    "selected_selection_score": finite_float(selected.get("selection_score")),
                    "selected_utility_base": finite_float(selected.get("utility_base")),
                    "selected_utility_delta": finite_float(selected.get("utility_delta")),
                    "selected_min_clearance_gt": finite_float(selected.get("min_clearance_gt")),
                    "oracle_id": int(oracle.get("id", -1)),
                    "oracle_type": oracle.get("type"),
                    "oracle_margin_gt": oracle_margin,
                    "oracle_margin_pred": finite_float(oracle.get("margin_pred")),
                    "oracle_selection_score": finite_float(oracle.get("selection_score")),
                    "oracle_utility_base": finite_float(oracle.get("utility_base")),
                    "oracle_utility_delta": finite_float(oracle.get("utility_delta")),
                    "oracle_min_clearance_gt": finite_float(oracle.get("min_clearance_gt")),
                    "oracle_gap_gt": gap,
                    "safe_missed": safe_missed,
                }
            )

    examples.sort(key=lambda item: (not item["safe_missed"], -(item["oracle_gap_gt"] or 0.0)))
    examples = examples[:max_examples]
    denom = max(valid_frames, 1)
    summary = {
        "total_rows": total_rows,
        "rows_with_candidates": rows_with_candidates,
        "valid_candidate_frames": valid_frames,
        "selected_rmvr_rate": selected_rmvr_frames / denom,
        "safe_candidate_available_rate": safe_available_frames / denom,
        "safe_candidate_missed_rate": safe_missed_frames / denom,
        "oracle_selected_rate": oracle_selected_frames / denom,
        "selected_collision_candidate_rate": collision_candidate_frames / denom,
        "mean_oracle_margin_gap_gt": sum(gaps) / len(gaps) if gaps else None,
        "max_oracle_margin_gap_gt": max(gaps) if gaps else None,
        "clearance_threshold": clearance_threshold,
    }
    return summary, examples


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: str, examples: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted({key for row in examples for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in examples:
            writer.writerow(row)


def write_md(path: str, method: str, summary: Dict[str, Any], examples: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        f"# Candidate Oracle Report: {method}",
        "",
        "This report compares the deployed selected candidate with the candidate-level GT reaction-margin oracle.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.6g} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Largest Missed-Oracle Examples", ""])
    if examples:
        lines.append("| depth_count | selected | selected margin | oracle | oracle margin | gap | safe missed |")
        lines.append("|---:|---|---:|---|---:|---:|---|")
        for ex in examples:
            lines.append(
                "| {depth_count} | {selected_type}#{selected_id} | {selected_margin_gt:.3f} | "
                "{oracle_type}#{oracle_id} | {oracle_margin_gt:.3f} | {oracle_gap_gt:.3f} | {safe_missed} |".format(**ex)
            )
    else:
        lines.append("No missed-oracle examples were found.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL after annotate_gt_reaction_margin --annotate-candidates")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-csv", default="")
    p.add_argument("--output-md", default="")
    p.add_argument("--method", default="")
    p.add_argument("--clearance-threshold", type=float, default=0.25)
    p.add_argument("--max-examples", type=int, default=40)
    return p


def main(args):
    rows = list(read_jsonl(args.input))
    summary, examples = summarize(rows, args.clearance_threshold, args.max_examples)
    method = args.method or method_from_path(args.input)
    payload = {"method": method, "input": args.input, "summary": summary, "examples": examples}
    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_csv:
        write_csv(args.output_csv, examples)
    if args.output_md:
        write_md(args.output_md, method, summary, examples)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parser().parse_args())
