import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List


SCENARIO_SPECS = {
    "blind_corner": "Hidden obstacle or narrow passage behind an L-shaped corner.",
    "doorway": "Occluder at a door frame with risk behind the threshold.",
    "t_junction": "Limited-FOV approach to a branch where risk is initially hidden.",
    "occluded_forest": "Dense obstacle field with a second obstacle hidden behind a visible one.",
    "limited_fov": "Repeated runs under different camera FOV settings.",
}

REQUIRED_RUN_FIELDS = (
    "time",
    "reaction_margin",
    "first_visible_time",
    "arrival_time_to_risk",
    "candidate_type",
    "speed",
    "inference_latency_ms",
    "emergency_brake",
)

OPTIONAL_GT_FIELDS = (
    "reaction_margin_gt",
    "reaction_window_gt",
    "rm_event_valid_gt",
    "rm_right_censored_gt",
    "rm_no_entry_gt",
    "risk_visible_at_t0_gt",
    "critical_risk_point_id",
    "critical_risk_weight",
    "calibrated_risk",
    "calibrated_risk_prob",
    "risk_upper_bound",
    "intervention_type",
    "intervention_reason",
    "reaction_budget_ms",
    "latency_violation",
    "first_visible_time_gt",
    "arrival_time_to_risk_gt",
    "critical_first_visible_time_gt",
    "critical_arrival_time_gt",
    "hidden_risk_gt",
    "valid_reaction_margin_gt",
    "selected_rmvr_gt",
    "min_clearance_gt",
    "collision_gt",
    "selected_traj_min_clearance_gt",
    "selected_traj_collision_gt",
    "min_clearance_exec",
    "mean_clearance_exec",
    "collision_exec",
    "success_exec",
    "timeout_exec",
    "path_time_exec",
    "mean_speed_exec",
    "goal_distance_final",
    "goal_distance_min",
    "reached_goal_exec",
    "first_arrival_time_exec",
    "first_collision_time_exec",
    "first_collision_goal_distance",
    "first_collision_clearance",
    "time_to_collision_exec",
    "motion_start_time_exec",
    "motion_start_delay_exec",
    "motion_start_goal_distance",
    "motion_start_speed_exec",
    "mean_speed_motion_exec",
    "first_collision_speed_exec",
    "first_arrival_speed_exec",
    "time_to_collision_from_motion_exec",
    "time_to_arrival_from_motion_exec",
    "time_to_terminal_from_motion_exec",
    "progress_at_collision_exec",
    "progress_at_terminal_exec",
    "distance_travelled_before_collision_exec",
    "distance_travelled_before_terminal_exec",
    "monitor_terminal_event",
    "monitor_terminal_time",
    "monitor_trimmed_at_terminal",
    "exec_rows_active",
    "exec_rows_untrimmed",
    "collision_exec_source",
    "success_exec_source",
    "evaluation_success_distance",
    "evaluation_collision_clearance",
    "planner_success_distance",
    "planner_arrival_distance",
)

TYPE_NAMES = ("progress", "probe", "brake", "yield")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def parse_rows(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        if isinstance(data, dict) and "rows" in data:
            return data["rows"]
        return [data]
    raise ValueError(f"Unsupported scenario log format: {path}")


def discover_logs(paths: Iterable[str]) -> List[str]:
    logs = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in files:
                    if os.path.splitext(name)[1].lower() in {".csv", ".json", ".jsonl"}:
                        logs.append(os.path.join(root, name))
        else:
            logs.append(path)
    return sorted(logs)


def scenario_from_path(path: str, default: str) -> str:
    lower = path.lower()
    for name in SCENARIO_SPECS:
        if name in lower:
            return name
    return default


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



def sigmoid(value):
    if value is None:
        return None
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def first_float(row, *keys):
    for key in keys:
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def risk_probability(row):
    prob = first_float(row, "calibrated_risk_prob", "calibrated_risk", "selected_risk_prob", "risk_prob")
    if prob is not None:
        return min(max(prob, 0.0), 1.0)
    return sigmoid(first_float(row, "selected_risk_logit", "risk_logit", "insufficient_margin_logit"))


def risk_upper_bound(row):
    value = first_float(row, "selected_risk_upper_bound", "risk_upper_bound", "calibrated_risk_upper_bound")
    return min(max(value, 0.0), 1.0) if value is not None else None


def intervention_type_name(row):
    value = row.get("intervention_type") or row.get("selected_intervention_type")
    if value not in (None, ""):
        return str(value).strip().lower()
    selected = candidate_type_name(row.get("candidate_type"))
    if selected == "probe":
        return "probe"
    if selected in {"brake", "yield"} or parse_bool(row.get("emergency_brake")):
        return "brake"
    return "unknown"


def latency_violation(row):
    explicit = row.get("latency_violation")
    if explicit not in (None, ""):
        return parse_bool(explicit)
    window = parse_float(row.get("reaction_window_gt"))
    budget_ms = parse_float(row.get("reaction_budget_ms"))
    if window is None or budget_ms is None:
        return None
    return window < budget_ms / 1000.0

def first_present(rows, key):
    for row in rows:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def bool_values(rows, key):
    return [parse_bool(row.get(key)) for row in rows if key in row and row.get(key) not in (None, "")]


def source_value(rows, key, fallback):
    value = first_present(rows, key)
    if value is None:
        return fallback
    return str(value)


def group_key_for_rows(rows, path, args):
    scenario = args.scenario or first_present(rows, "scenario") or scenario_from_path(path, args.default_scenario)
    method = args.method or first_present(rows, "method")
    parts = []
    if args.group_by_method and method:
        parts.append(str(method))
    parts.append(str(scenario))
    if args.group_by_success_distance:
        success_distance = parse_float(
            first_present(rows, "evaluation_success_distance"),
            parse_float(first_present(rows, "monitor_success_distance")),
        )
        if success_distance is not None:
            parts.append(f"eval_success_{success_distance:g}m")
    return "__".join(parts)


def summarize_run(rows: List[Dict]) -> Dict:
    if not rows:
        return {}

    times = [parse_float(row.get("time")) for row in rows]
    times = [t for t in times if t is not None]
    pred_margins = [parse_float(row.get("selected_margin_pred", row.get("reaction_margin"))) for row in rows]
    pred_margins = [m for m in pred_margins if m is not None]
    gt_margins = [parse_float(row.get("reaction_margin_gt")) for row in rows]
    gt_margins = [m for m in gt_margins if m is not None]
    hidden_gt_margins = [
        parse_float(row.get("reaction_margin_gt"))
        for row in rows
        if parse_bool(row.get("hidden_risk_gt"))
    ]
    hidden_gt_margins = [m for m in hidden_gt_margins if m is not None]
    margins = gt_margins if gt_margins else pred_margins
    first_visible = [parse_float(row.get("first_visible_time_gt", row.get("first_visible_time"))) for row in rows]
    first_visible = [v for v in first_visible if v is not None]
    arrival = [parse_float(row.get("arrival_time_to_risk_gt", row.get("arrival_time_gt", row.get("arrival_time_to_risk")))) for row in rows]
    arrival = [v for v in arrival if v is not None]
    speeds = [parse_float(row.get("speed")) for row in rows]
    speeds = [v for v in speeds if v is not None]
    latency = [parse_float(row.get("inference_latency_ms")) for row in rows]
    latency = [v for v in latency if v is not None]
    selected_clearances = [
        parse_float(row.get("selected_traj_min_clearance_gt", row.get("min_clearance_gt"))) for row in rows
    ]
    selected_clearances = [v for v in selected_clearances if v is not None]
    exec_min_clearance = parse_float(first_present(rows, "min_clearance_exec"))
    exec_mean_clearance = parse_float(first_present(rows, "mean_clearance_exec"))
    path_time_exec = parse_float(first_present(rows, "path_time_exec"))
    mean_speed_exec = parse_float(first_present(rows, "mean_speed_exec"))
    goal_distance_final = parse_float(first_present(rows, "goal_distance_final"))
    goal_distance_min = parse_float(first_present(rows, "goal_distance_min"))
    first_arrival_time_exec = parse_float(first_present(rows, "first_arrival_time_exec"))
    first_collision_time_exec = parse_float(first_present(rows, "first_collision_time_exec"))
    first_collision_goal_distance = parse_float(first_present(rows, "first_collision_goal_distance"))
    first_collision_clearance = parse_float(first_present(rows, "first_collision_clearance"))
    time_to_collision_exec = parse_float(first_present(rows, "time_to_collision_exec"))
    motion_start_time_exec = parse_float(first_present(rows, "motion_start_time_exec"))
    motion_start_delay_exec = parse_float(first_present(rows, "motion_start_delay_exec"))
    motion_start_goal_distance = parse_float(first_present(rows, "motion_start_goal_distance"))
    motion_start_speed_exec = parse_float(first_present(rows, "motion_start_speed_exec"))
    mean_speed_motion_exec = parse_float(first_present(rows, "mean_speed_motion_exec"))
    first_collision_speed_exec = parse_float(first_present(rows, "first_collision_speed_exec"))
    first_arrival_speed_exec = parse_float(first_present(rows, "first_arrival_speed_exec"))
    time_to_collision_from_motion_exec = parse_float(first_present(rows, "time_to_collision_from_motion_exec"))
    time_to_arrival_from_motion_exec = parse_float(first_present(rows, "time_to_arrival_from_motion_exec"))
    time_to_terminal_from_motion_exec = parse_float(first_present(rows, "time_to_terminal_from_motion_exec"))
    progress_at_collision_exec = parse_float(first_present(rows, "progress_at_collision_exec"))
    progress_at_terminal_exec = parse_float(first_present(rows, "progress_at_terminal_exec"))
    distance_travelled_before_collision_exec = parse_float(first_present(rows, "distance_travelled_before_collision_exec"))
    distance_travelled_before_terminal_exec = parse_float(first_present(rows, "distance_travelled_before_terminal_exec"))
    monitor_terminal_time = parse_float(first_present(rows, "monitor_terminal_time"))
    exec_rows_active = parse_float(first_present(rows, "exec_rows_active"))
    exec_rows_untrimmed = parse_float(first_present(rows, "exec_rows_untrimmed"))
    candidate_types = [candidate_type_name(row.get("candidate_type")) for row in rows]
    collision_exec_values = bool_values(rows, "collision_exec")
    success_exec_values = bool_values(rows, "success_exec")
    selected_collision_values = bool_values(rows, "selected_traj_collision_gt") or bool_values(rows, "collision_gt")
    risk_probs = [risk_probability(row) for row in rows]
    risk_uppers = [risk_upper_bound(row) for row in rows]
    intervention_types = [intervention_type_name(row) for row in rows]
    intervention_known = [t for t in intervention_types if t != "unknown"]
    latency_violations = [latency_violation(row) for row in rows]
    latency_violations = [v for v in latency_violations if v is not None]
    intervention_pairs = []
    for row, intervention_type in zip(rows, intervention_types):
        safe_value = row.get("yopo_top1_safe_gt", row.get("yopo_actually_safe_gt"))
        if safe_value not in (None, ""):
            safe = parse_bool(safe_value)
        else:
            margin = parse_float(row.get("reaction_margin_gt"))
            if margin is None:
                continue
            safe = margin >= 0.0
        intervention_pairs.append((intervention_type not in {"keep", "unknown"}, safe))
    false_interventions = [float(intervened and safe) for intervened, safe in intervention_pairs]
    missed_interventions = [float((not intervened) and (not safe)) for intervened, safe in intervention_pairs]
    risk_covered = [parse_bool(row.get("risk_covered")) for row in rows if row.get("risk_covered") not in (None, "")]
    legacy_collision_values = bool_values(rows, "collision") or bool_values(rows, "collision_flag")
    legacy_success_values = bool_values(rows, "success") or bool_values(rows, "success_flag")
    collision_exec = any(collision_exec_values) if collision_exec_values else any(legacy_collision_values)
    success_exec = any(success_exec_values) if success_exec_values else any(legacy_success_values)
    collision_source = source_value(
        rows,
        "collision_exec_source",
        "execution_monitor" if collision_exec_values else "legacy_ros_collision_field",
    )
    success_source = source_value(
        rows,
        "success_exec_source",
        "execution_monitor" if success_exec_values else "legacy_ros_success_field",
    )
    execution_available = bool(collision_exec_values or success_exec_values or exec_min_clearance is not None)
    warnings = []
    if not execution_available:
        warnings.append("missing_execution_metrics")
    if collision_source in {"reference_or_planner_start_to_gt_pointcloud", "legacy_ros_collision_field"}:
        warnings.append(f"non_odom_collision_source:{collision_source}")

    summary = {
        "method": source_value(rows, "method", ""),
        "run_id": source_value(rows, "run_id", ""),
        "goal_segment_id": source_value(rows, "goal_segment_id", ""),
        "evaluation_success_distance": parse_float(
            first_present(rows, "evaluation_success_distance"),
            parse_float(first_present(rows, "monitor_success_distance")),
        ),
        "evaluation_collision_clearance": parse_float(
            first_present(rows, "evaluation_collision_clearance"),
            parse_float(first_present(rows, "monitor_collision_clearance")),
        ),
        "planner_success_distance": parse_float(
            first_present(rows, "planner_success_distance"),
            parse_float(first_present(rows, "success_distance")),
        ),
        "planner_arrival_distance": parse_float(
            first_present(rows, "planner_arrival_distance"),
            parse_float(first_present(rows, "arrival_distance")),
        ),
        "success_exec": float(success_exec),
        "collision_exec": float(collision_exec),
        "success": float(success_exec),
        "collision": float(collision_exec),
        "success_metric_source": success_source,
        "collision_metric_source": collision_source,
        "path_time": path_time_exec if path_time_exec is not None else ((max(times) - min(times)) if times else None),
        "path_time_exec": path_time_exec,
        "sample_count": len(rows),
        "reaction_margin_violation_rate": mean([float(m < 0.0) for m in margins]),
        "rm_violation_rate": mean([float(m < 0.0) for m in margins]),
        "calibrated_risk_mean": mean([p for p in risk_probs if p is not None]),
        "risk_upper_bound_mean": mean([p for p in risk_uppers if p is not None]),
        "latency_violation_rate": mean([float(v) for v in latency_violations]),
        "risk_coverage": mean([float(v) for v in risk_covered]),
        "keep_rate": mean([float(t == "keep") for t in intervention_known]),
        "rerank_rate": mean([float(t == "rerank") for t in intervention_known]),
        "probe_rate": mean([float(t == "probe") for t in intervention_known]),
        "brake_rate": mean([float(t == "brake") for t in intervention_known]),
        "false_intervention_rate": mean(false_interventions),
        "missed_intervention_rate": mean(missed_interventions),
        "rmvr_source": 1.0 if gt_margins else 0.0,
        "gt_rmvr_valid_count": len(gt_margins),
        "gt_rmvr_coverage": len(gt_margins) / max(len(rows), 1),
        "selected_rmvr_gt": mean([float(m < 0.0) for m in gt_margins]),
        "hidden_risk_gt_count": len(hidden_gt_margins),
        "hidden_risk_gt_coverage": len(hidden_gt_margins) / max(len(rows), 1),
        "selected_rmvr_gt_hidden": mean([float(m < 0.0) for m in hidden_gt_margins]),
        "mean_reaction_margin_gt_hidden": mean(hidden_gt_margins),
        "minimum_reaction_margin_gt_hidden": min(hidden_gt_margins) if hidden_gt_margins else None,
        "predicted_reaction_margin_violation_rate": mean([float(m < 0.0) for m in pred_margins]),
        "minimum_reaction_margin": min(margins) if margins else None,
        "mean_reaction_margin": mean(margins),
        "minimum_reaction_margin_gt": min(gt_margins) if gt_margins else None,
        "mean_reaction_margin_gt": mean(gt_margins),
        "mean_predicted_reaction_margin": mean(pred_margins),
        "first_visible_time_mean": mean(first_visible),
        "arrival_time_to_risk_mean": mean(arrival),
        "mean_speed": mean_speed_exec if mean_speed_exec is not None else mean(speeds),
        "mean_speed_exec": mean_speed_exec,
        "inference_latency_ms_mean": mean(latency),
        "min_clearance_exec": exec_min_clearance,
        "mean_clearance_exec": exec_mean_clearance,
        "goal_distance_final": goal_distance_final,
        "goal_distance_min": goal_distance_min,
        "reached_goal_exec": float(any(bool_values(rows, "reached_goal_exec"))) if bool_values(rows, "reached_goal_exec") else None,
        "first_arrival_time_exec": first_arrival_time_exec,
        "first_collision_time_exec": first_collision_time_exec,
        "first_collision_goal_distance": first_collision_goal_distance,
        "first_collision_clearance": first_collision_clearance,
        "time_to_collision_exec": time_to_collision_exec,
        "motion_start_time_exec": motion_start_time_exec,
        "motion_start_delay_exec": motion_start_delay_exec,
        "motion_start_goal_distance": motion_start_goal_distance,
        "motion_start_speed_exec": motion_start_speed_exec,
        "mean_speed_motion_exec": mean_speed_motion_exec,
        "first_collision_speed_exec": first_collision_speed_exec,
        "first_arrival_speed_exec": first_arrival_speed_exec,
        "time_to_collision_from_motion_exec": time_to_collision_from_motion_exec,
        "time_to_arrival_from_motion_exec": time_to_arrival_from_motion_exec,
        "time_to_terminal_from_motion_exec": time_to_terminal_from_motion_exec,
        "progress_at_collision_exec": progress_at_collision_exec,
        "progress_at_terminal_exec": progress_at_terminal_exec,
        "distance_travelled_before_collision_exec": distance_travelled_before_collision_exec,
        "distance_travelled_before_terminal_exec": distance_travelled_before_terminal_exec,
        "monitor_terminal_time": monitor_terminal_time,
        "monitor_terminal_event": source_value(rows, "monitor_terminal_event", ""),
        "monitor_trimmed_at_terminal": float(any(bool_values(rows, "monitor_trimmed_at_terminal"))) if bool_values(rows, "monitor_trimmed_at_terminal") else None,
        "exec_rows_active": exec_rows_active,
        "exec_rows_untrimmed": exec_rows_untrimmed,
        "selected_traj_collision_gt": mean([float(value) for value in selected_collision_values]),
        "selected_traj_min_clearance_gt": min(selected_clearances) if selected_clearances else None,
        "selected_traj_mean_clearance_gt": mean(selected_clearances),
        "min_clearance_gt": min(selected_clearances) if selected_clearances else None,
        "mean_clearance_gt": mean(selected_clearances),
        "execution_metrics_available": float(execution_available),
        "exec_metrics_valid": float(len(warnings) == 0),
        "benchmark_warnings": ";".join(warnings),
        "emergency_brake_rate": mean([float(parse_bool(row.get("emergency_brake"))) for row in rows]),
    }
    for type_name in TYPE_NAMES:
        summary[f"selected_{type_name}_rate"] = mean([float(t == type_name) for t in candidate_types])
    return summary


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def dedupe_run_summaries(run_summaries: List[Dict]) -> List[Dict]:
    deduped = []
    seen = set()
    for idx, summary in enumerate(run_summaries):
        method = summary.get("method") or ""
        run_id = summary.get("run_id") or f"__file_index_{idx}"
        goal_segment_id = summary.get("goal_segment_id") or ""
        key = (str(method), str(run_id), str(goal_segment_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(summary)
    return deduped


def aggregate(run_summaries: List[Dict]) -> Dict:
    if not run_summaries:
        return {"run_count": 0}
    input_count = len(run_summaries)
    run_summaries = dedupe_run_summaries(run_summaries)
    keys = sorted(set().union(*(summary.keys() for summary in run_summaries)))
    out = {"run_count": len(run_summaries), "input_log_count": input_count, "duplicate_run_count": input_count - len(run_summaries)}
    for key in keys:
        if key == "sample_count":
            out[key] = int(sum(summary.get(key, 0) for summary in run_summaries))
        elif key in {"success", "collision", "success_exec", "collision_exec"}:
            out[f"{key}_rate"] = mean(summary.get(key, 0.0) for summary in run_summaries)
        elif any(isinstance(summary.get(key), str) for summary in run_summaries if key in summary):
            values = sorted({str(summary.get(key)) for summary in run_summaries if summary.get(key)})
            out[key] = ",".join(values)
        else:
            out[key] = mean(summary.get(key) for summary in run_summaries)
    return out


def write_csv(path: str, summaries: Dict[str, Dict]):
    keys = sorted(set().union(*(metrics.keys() for metrics in summaries.values())))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", *keys])
        writer.writeheader()
        for scenario, metrics in sorted(summaries.items()):
            writer.writerow({"scenario": scenario, **metrics})


def benchmark(args):
    if args.print_manifest:
        print(
            json.dumps(
                {
                    "scenarios": SCENARIO_SPECS,
                    "required_fields": REQUIRED_RUN_FIELDS,
                    "optional_gt_fields": OPTIONAL_GT_FIELDS,
                },
                indent=2,
            )
        )
        return

    logs = discover_logs(args.logs)
    grouped = defaultdict(list)
    for path in logs:
        rows = parse_rows(path)
        group_key = group_key_for_rows(rows, path, args)
        grouped[group_key].append(summarize_run(rows))
    invalid = [
        f"{scenario}[run={idx}]: {summary.get('benchmark_warnings')}"
        for scenario, summaries in grouped.items()
        for idx, summary in enumerate(summaries)
        if summary.get("benchmark_warnings")
    ]
    if args.require_exec_metrics and invalid:
        raise ValueError("Invalid execution metrics for paper benchmark: " + " | ".join(invalid))

    summaries = {scenario: aggregate(runs) for scenario, runs in grouped.items()}
    output = {
        "benchmark": "OARM occlusion scenario benchmark",
        "online_inputs": ["depth", "state", "goal"],
        "mapless_online_inference": True,
        "scenario_specs": SCENARIO_SPECS,
        "warnings": invalid,
        "metrics": summaries,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True)
    if args.csv_output:
        write_csv(args.csv_output, summaries)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", help="scenario CSV/JSON/JSONL logs or directories containing them")
    p.add_argument("--scenario", type=str, default="", help="force all logs to one scenario name")
    p.add_argument("--method", type=str, default="", help="force all logs to one method name")
    p.add_argument("--group-by-method", action="store_true", default=True, help="group benchmark rows by method and scenario")
    p.add_argument("--no-group-by-method", dest="group_by_method", action="store_false")
    p.add_argument("--group-by-success-distance", action="store_true", default=True, help="keep 1m/2m evaluator thresholds as separate benchmark groups")
    p.add_argument("--no-group-by-success-distance", dest="group_by_success_distance", action="store_false")
    p.add_argument("--default-scenario", type=str, default="unknown")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--csv-output", type=str, default="")
    p.add_argument("--print-manifest", action="store_true")
    p.add_argument("--require-exec-metrics", action="store_true", help="fail if a run lacks odom-based execution metrics")
    return p


if __name__ == "__main__":
    benchmark(parser().parse_args())
