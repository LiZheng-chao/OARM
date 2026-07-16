import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def first_present(rows: Iterable[Dict], key: str):
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def candidate_type_name(row: Dict) -> str:
    value = row.get("candidate_type", row.get("selected_type", "unknown"))
    if value is None or value == "":
        return "unknown"
    text = str(value).strip().lower()
    if text.isdigit():
        names = ("progress", "probe", "brake", "yield")
        idx = int(text)
        if 0 <= idx < len(names):
            return names[idx]
    if text == "backup":
        return "yield"
    return text


def label_for(path: str, rows: List[Dict], group_by_type: bool, row: Dict = None) -> Tuple[str, str, str]:
    method = first_present(rows, "method") or os.path.splitext(os.path.basename(path))[0]
    scenario = first_present(rows, "scenario") or os.path.basename(os.path.dirname(path))
    type_name = candidate_type_name(row) if group_by_type and row is not None else "all"
    return str(method), str(scenario), type_name


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_mean = mean([x for x, _y in pairs])
    y_mean = mean([y for _x, y in pairs])
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x, _y in pairs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for _x, y in pairs))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def summarize(method: str, scenario: str, type_name: str, rows: List[Dict]) -> Dict:
    pred = [parse_float(row.get("selected_margin_pred", row.get("reaction_margin"))) for row in rows]
    gt = [parse_float(row.get("reaction_margin_gt")) for row in rows]
    valid_pairs = [(p, g) for p, g in zip(pred, gt) if p is not None and g is not None]
    errors = [p - g for p, g in valid_pairs]
    abs_errors = [abs(e) for e in errors]
    sign_matches = [float((p >= 0.0) == (g >= 0.0)) for p, g in valid_pairs]
    censored = [parse_bool(row.get("reaction_margin_censored_gt")) for row in rows if "reaction_margin_censored_gt" in row]
    valid_flags = [parse_bool(row.get("valid_reaction_margin_gt")) for row in rows if "valid_reaction_margin_gt" in row]
    return {
        "method": method,
        "scenario": scenario,
        "candidate_type": type_name,
        "sample_count": len(rows),
        "valid_pair_count": len(valid_pairs),
        "margin_valid_rate": mean([float(v) for v in valid_flags]),
        "margin_censored_rate": mean([float(v) for v in censored]),
        "margin_prediction_mean": mean([p for p, _g in valid_pairs]),
        "margin_gt_mean": mean([g for _p, g in valid_pairs]),
        "margin_prediction_bias": mean(errors),
        "margin_prediction_mae": mean(abs_errors),
        "margin_prediction_rmse": math.sqrt(mean([e * e for e in errors])) if errors else None,
        "margin_prediction_corr": corr([p for p, _g in valid_pairs], [g for _p, g in valid_pairs]),
        "margin_sign_accuracy": mean(sign_matches),
        "predicted_negative_rate": mean([float(p < 0.0) for p, _g in valid_pairs]),
        "gt_negative_rate": mean([float(g < 0.0) for _p, g in valid_pairs]),
        "selected_traj_collision_rate": mean(
            [
                float(parse_bool(row.get("selected_traj_collision_gt", row.get("collision_gt"))))
                for row in rows
                if row.get("selected_traj_collision_gt", row.get("collision_gt")) not in (None, "")
            ]
        ),
    }


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(args):
    grouped = defaultdict(list)
    for path in args.logs:
        rows = read_jsonl(path)
        for row in rows:
            key = label_for(path, rows, args.group_by_type, row)
            grouped[key].append(row)
    summaries = [
        summarize(method, scenario, type_name, rows)
        for (method, scenario, type_name), rows in sorted(grouped.items())
    ]
    output = {"logs": args.logs, "summaries": summaries}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True)
    if args.csv_output:
        write_csv(args.csv_output, summaries)
    print(json.dumps(output, indent=2, sort_keys=True))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="GT-annotated or execution-monitor JSONL logs")
    p.add_argument("--group-by-type", action="store_true")
    p.add_argument("--output", default="")
    p.add_argument("--csv-output", default="")
    return p


if __name__ == "__main__":
    diagnostics(parser().parse_args())
