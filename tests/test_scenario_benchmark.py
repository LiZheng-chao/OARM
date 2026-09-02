import json
from types import SimpleNamespace

from OARM.eval import annotate_gt_reaction_margin
from OARM.eval import execution_monitor
from OARM.eval import scenario_benchmark


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_execution_monitor_accepts_fixed_pointcloud_pattern():
    args = execution_monitor.parser().parse_args(
        ["--input", "planner.jsonl", "--output", "monitored.jsonl", "--pointcloud-pattern", "pointcloud-0.ply"]
    )

    assert args.pointcloud_pattern == "pointcloud-0.ply"

    annotation_args = annotate_gt_reaction_margin.parser().parse_args(
        [
            "--input",
            "planner.jsonl",
            "--output",
            "annotated.jsonl",
            "--pointcloud-pattern",
            "pointcloud-0.ply",
            "--map-id",
            "0",
            "--force-map-id",
            "--use-esdf-los",
        ]
    )
    assert annotation_args.force_map_id
    assert annotation_args.map_id == 0
    assert annotation_args.pointcloud_pattern == "pointcloud-0.ply"
    budget_s, source = annotate_gt_reaction_margin.reaction_budget_for_row(
        {"reaction_budget_ms": 547.5, "tau_total_s": 0.9},
        0.35,
    )
    assert abs(budget_s - 0.5475) < 1e-9
    assert source == "reaction_budget_ms"
    assert annotate_gt_reaction_margin.reaction_budget_for_row({}, 0.35) == (0.35, "config_reaction_time")


def test_benchmark_prefers_monitored_log_for_same_run(tmp_path):
    identity = {
        "method": "oarm",
        "run_id": "run-1",
        "goal_segment_id": 0,
        "scenario": "ceiling_gate",
    }
    write_jsonl(
        tmp_path / "planner.jsonl",
        [{**identity, "time": 1.0, "collision": False, "success": True}],
    )
    write_jsonl(
        tmp_path / "exec.jsonl",
        [{**identity, "time": 1.0, "exec_log_source": "odom_callback", "success": True}],
    )
    write_jsonl(
        tmp_path / "monitored.jsonl",
        [
            {
                **identity,
                "time": 1.0,
                "collision_exec": False,
                "success_exec": True,
                "collision_exec_source": "executed_odom_to_gt_pointcloud",
                "success_exec_source": "first_terminal_event_monitor",
                "min_clearance_exec": 0.5,
            }
        ],
    )
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(
        logs=[str(tmp_path)],
        scenario="",
        method="",
        group_by_method=True,
        group_by_success_distance=False,
        default_scenario="unknown",
        output=str(output),
        csv_output=str(tmp_path / "benchmark.csv"),
        print_manifest=False,
        require_exec_metrics=True,
    )

    scenario_benchmark.benchmark(args)

    result = json.loads(output.read_text(encoding="utf-8"))
    metrics = result["metrics"]["oarm__ceiling_gate"]
    assert metrics["run_count"] == 1
    assert metrics["input_log_count"] == 1
    assert metrics["success_exec_rate"] == 1.0
    assert metrics["exec_metrics_valid"] == 1.0
