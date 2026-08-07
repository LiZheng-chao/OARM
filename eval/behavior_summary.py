import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_benchmark(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    metrics = payload.get('metrics', payload if isinstance(payload, dict) else {})
    by_method: Dict[str, Dict[str, Any]] = {}
    for _key, row in metrics.items():
        if not isinstance(row, dict):
            continue
        method = row.get('method')
        if method:
            by_method[str(method)] = row
    return by_method


def exec_metrics(path: Path, goal_z: float, nominal_z: float, ceiling_z: Optional[float]) -> Dict[str, Any]:
    rows = list(read_jsonl(path))
    pos = [r.get('odom_pos_w') for r in rows if isinstance(r.get('odom_pos_w'), list) and len(r.get('odom_pos_w')) >= 3]
    speeds = [finite_float(r.get('speed')) for r in rows]
    speeds = [v for v in speeds if v is not None]
    goal_distances = [finite_float(r.get('goal_distance')) for r in rows]
    goal_distances = [v for v in goal_distances if v is not None]
    z = [finite_float(p[2]) for p in pos]
    z = [v for v in z if v is not None]
    x = [finite_float(p[0]) for p in pos]
    x = [v for v in x if v is not None]
    y = [finite_float(p[1]) for p in pos]
    y = [v for v in y if v is not None]
    if len(rows) >= 2:
        t0 = finite_float(rows[0].get('time')) or 0.0
        t1 = finite_float(rows[-1].get('time')) or 0.0
        duration = max(0.0, t1 - t0)
    else:
        duration = None
    vertical_abs = [abs(v - nominal_z) for v in z]
    vertical_goal_abs = [abs(v - goal_z) for v in z]
    ceiling_margin_min = None
    ceiling_violation_rate = None
    if ceiling_z is not None and z:
        margins = [ceiling_z - v for v in z]
        ceiling_margin_min = min(margins)
        ceiling_violation_rate = sum(v > ceiling_z for v in z) / len(z)
    return {
        'exec_rows': len(rows),
        'duration_s_exec_log': duration,
        'success_any_exec_log': any(bool(r.get('success') or r.get('success_flag') or r.get('arrive')) for r in rows),
        'min_goal_distance_exec_log': min(goal_distances) if goal_distances else None,
        'last_goal_distance_exec_log': goal_distances[-1] if goal_distances else None,
        'mean_speed_exec_log': sum(speeds) / len(speeds) if speeds else None,
        'max_speed_exec_log': max(speeds) if speeds else None,
        'z_min': min(z) if z else None,
        'z_max': max(z) if z else None,
        'z_mean': sum(z) / len(z) if z else None,
        'vertical_detour_max_from_nominal': max(vertical_abs) if vertical_abs else None,
        'vertical_detour_mean_from_nominal': sum(vertical_abs) / len(vertical_abs) if vertical_abs else None,
        'vertical_detour_max_from_goal': max(vertical_goal_abs) if vertical_goal_abs else None,
        'vertical_detour_mean_from_goal': sum(vertical_goal_abs) / len(vertical_goal_abs) if vertical_goal_abs else None,
        'y_abs_max': max(abs(v) for v in y) if y else None,
        'x_progress_final': x[-1] - x[0] if len(x) >= 2 else None,
        'ceiling_z': ceiling_z,
        'ceiling_margin_min': ceiling_margin_min,
        'ceiling_violation_rate': ceiling_violation_rate,
    }


def merge(method: str, exec_path: Path, benchmark: Dict[str, Dict[str, Any]], args) -> Dict[str, Any]:
    row = {'method': method, 'exec_log': str(exec_path)}
    row.update(exec_metrics(exec_path, args.goal_z, args.nominal_z, args.ceiling_z))
    bench = benchmark.get(method, {})
    for key in (
        'success_rate',
        'success_exec_rate',
        'collision_rate',
        'collision_exec_rate',
        'path_time_exec',
        'mean_speed_exec',
        'min_clearance_exec',
        'min_clearance_gt',
        'selected_rmvr_gt',
        'selected_rmvr_gt_hidden',
        'selected_traj_min_clearance_gt',
        'selected_traj_mean_clearance_gt',
        'first_visible_time_mean',
        'mean_reaction_margin_gt',
        'goal_distance_min',
        'goal_distance_final',
        'monitor_terminal_event',
    ):
        if key in bench:
            row[key] = bench[key]
    return row


def parse_run(value: str):
    if '=' not in value:
        raise argparse.ArgumentTypeError('--run expects METHOD=EXEC_JSONL')
    method, path = value.split('=', 1)
    method = method.strip()
    if not method or not path:
        raise argparse.ArgumentTypeError('--run expects non-empty METHOD=EXEC_JSONL')
    return method, Path(path)


def write_outputs(rows, args):
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'rows': rows}, indent=2, sort_keys=True), encoding='utf-8')
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with out.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            'method', 'success_rate', 'collision_rate', 'min_clearance_exec',
            'selected_rmvr_gt', 'path_time_exec', 'mean_speed_exec', 'z_max',
            'vertical_detour_max_from_goal', 'ceiling_margin_min', 'min_goal_distance_exec_log'
        ]
        lines = ['# OARM Behavior Summary', '', '| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |']
        for row in rows:
            vals = []
            for key in keys:
                value = row.get(key)
                if isinstance(value, float):
                    vals.append(f'{value:.4g}')
                elif value is None:
                    vals.append('')
                else:
                    vals.append(str(value))
            lines.append('| ' + ' | '.join(vals) + ' |')
        out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark-json', type=Path, default=None)
    p.add_argument('--run', action='append', type=parse_run, required=True, help='METHOD=EXEC_JSONL; repeat for each method')
    p.add_argument('--goal-z', type=float, default=2.0)
    p.add_argument('--nominal-z', type=float, default=2.0)
    p.add_argument('--ceiling-z', type=float, default=None)
    p.add_argument('--output-json', default='')
    p.add_argument('--output-csv', default='')
    p.add_argument('--output-md', default='')
    return p


def main():
    args = parser().parse_args()
    benchmark = read_benchmark(args.benchmark_json) if args.benchmark_json else {}
    rows = [merge(method, path, benchmark, args) for method, path in args.run]
    write_outputs(rows, args)
    print(json.dumps({'rows': rows}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
