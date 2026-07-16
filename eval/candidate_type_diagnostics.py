import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List


TYPE_NAMES = ("progress", "probe", "brake", "yield")


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


def parse_list(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, list) else None


def candidate_type_name(value):
    if value is None or value == "":
        return "unknown"
    text = str(value).strip().lower()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(TYPE_NAMES):
            return TYPE_NAMES[idx]
    if text == "backup":
        return "yield"
    return text


def first_present(rows: Iterable[Dict], key: str):
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


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


def signed_accuracy(preds, gts):
    pairs = [(p, g) for p, g in zip(preds, gts) if p is not None and g is not None]
    if not pairs:
        return None
    return mean([float((p >= 0.0) == (g >= 0.0)) for p, g in pairs])


def row_label(rows: List[Dict], path: str):
    method = first_present(rows, "method") or os.path.splitext(os.path.basename(path))[0]
    scenario = first_present(rows, "scenario") or os.path.basename(os.path.dirname(path))
    return str(method), str(scenario)


def selected_record(row: Dict, method: str, scenario: str):
    type_name = candidate_type_name(row.get("candidate_type", row.get("selected_type")))
    return {
        "method": method,
        "scenario": scenario,
        "source": "selected",
        "candidate_type": type_name,
        "pred_margin": parse_float(row.get("selected_margin_pred", row.get("reaction_margin"))),
        "gt_margin": parse_float(row.get("reaction_margin_gt")),
        "utility": parse_float(row.get("selected_utility", row.get("selected_selection_score"))),
        "traj_collision": parse_bool(row.get("selected_traj_collision_gt", row.get("collision_gt"))),
        "traj_min_clearance": parse_float(row.get("selected_traj_min_clearance_gt", row.get("min_clearance_gt"))),
        "progress_rate": parse_float(row.get("selected_progress_rate")),
        "goal_distance_drop": parse_float(row.get("selected_goal_distance_drop")),
        "goal_distance_drop_rate": parse_float(row.get("selected_goal_distance_drop_rate")),
        "emergency_brake": parse_bool(row.get("emergency_brake")),
    }


def aligned(values, length):
    if values is None or len(values) != length:
        return [None] * length
    return values


def all_candidate_records(row: Dict, method: str, scenario: str):
    types = None
    for key in ("candidate_types", "candidate_type_list", "all_candidate_types"):
        types = parse_list(row.get(key))
        if types:
            break
    if not types:
        return []
    n = len(types)
    pred = aligned(
        parse_list(row.get("candidate_margin_preds"))
        or parse_list(row.get("candidate_pred_margins"))
        or parse_list(row.get("candidate_reaction_margins")),
        n,
    )
    gt = aligned(parse_list(row.get("candidate_margin_gts")) or parse_list(row.get("candidate_gt_margins")), n)
    utility = aligned(parse_list(row.get("candidate_utilities")) or parse_list(row.get("candidate_scores")), n)
    collisions = aligned(parse_list(row.get("candidate_collision_gts")) or parse_list(row.get("candidate_collisions")), n)
    selected_id = row.get("selected_id")
    records = []
    for idx, type_value in enumerate(types):
        records.append(
            {
                "method": method,
                "scenario": scenario,
                "source": "all_candidates",
                "candidate_type": candidate_type_name(type_value),
                "pred_margin": parse_float(pred[idx]),
                "gt_margin": parse_float(gt[idx]),
                "utility": parse_float(utility[idx]),
                "traj_collision": parse_bool(collisions[idx]) if collisions[idx] is not None else None,
                "traj_min_clearance": None,
                "progress_rate": None,
                "goal_distance_drop": None,
                "goal_distance_drop_rate": None,
                "emergency_brake": None,
                "selected": str(selected_id) == str(idx) if selected_id not in (None, "") else None,
            }
        )
    return records


def summarize(records: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    total_by_parent = defaultdict(int)
    for record in records:
        parent = (record["method"], record["scenario"], record["source"])
        key = (*parent, record["candidate_type"])
        grouped[key].append(record)
        total_by_parent[parent] += 1

    rows = []
    for (method, scenario, source, type_name), group in sorted(grouped.items()):
        preds = [row["pred_margin"] for row in group]
        gts = [row["gt_margin"] for row in group]
        selected_values = [row.get("selected") for row in group if row.get("selected") is not None]
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "source": source,
                "candidate_type": type_name,
                "candidate_count": len(group),
                "type_rate": len(group) / max(total_by_parent[(method, scenario, source)], 1),
                "selected_rate": mean([float(v) for v in selected_values]),
                "mean_gt_margin": mean(gts),
                "mean_pred_margin": mean(preds),
                "mean_margin_bias": mean(
                    [p - g for p, g in zip(preds, gts) if p is not None and g is not None]
                ),
                "margin_corr": corr(preds, gts),
                "margin_sign_accuracy": signed_accuracy(preds, gts),
                "mean_utility": mean([row["utility"] for row in group]),
                "selected_traj_collision_rate": mean(
                    [float(row["traj_collision"]) for row in group if row["traj_collision"] is not None]
                ),
                "mean_traj_min_clearance": mean([row["traj_min_clearance"] for row in group]),
                "mean_progress_rate": mean([row["progress_rate"] for row in group]),
                "mean_goal_distance_drop": mean([row["goal_distance_drop"] for row in group]),
                "mean_goal_distance_drop_rate": mean([row["goal_distance_drop_rate"] for row in group]),
                "emergency_brake_rate": mean(
                    [float(row["emergency_brake"]) for row in group if row["emergency_brake"] is not None]
                ),
            }
        )
    return rows


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
    records = []
    all_candidate_rows = 0
    for path in args.logs:
        rows = read_jsonl(path)
        if not rows:
            continue
        method, scenario = row_label(rows, path)
        for row in rows:
            records.append(selected_record(row, method, scenario))
            all_records = all_candidate_records(row, method, scenario)
            all_candidate_rows += int(bool(all_records))
            records.extend(all_records)
    summaries = summarize(records)
    output = {
        "logs": args.logs,
        "record_count": len(records),
        "rows_with_all_candidate_arrays": all_candidate_rows,
        "note": "source=selected uses deployed selected candidates; source=all_candidates appears only when logs contain full candidate arrays.",
        "summaries": summaries,
    }
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
    p.add_argument("--output", default="")
    p.add_argument("--csv-output", default="")
    return p


if __name__ == "__main__":
    diagnostics(parser().parse_args())
