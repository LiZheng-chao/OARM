import argparse
import glob
import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_KEYS = [
    "total_loss",
    "margin_loss",
    "ranking_loss",
    "pairwise_ranking_accuracy",
    "ranking_pair_rate",
    "reaction_margin_gt_source_rate",
    "reaction_margin_proxy_source_rate",
    "candidate_hidden_risk_gt_rate",
    "generated_margin_valid_rate",
    "generated_margin_violation_rate",
    "risk_weight_nonzero_rate",
    "risk_weight_sum_mean",
    "mean_time",
    "risk_loss",
]


def latest_scalar(event_accumulator, tag):
    scalars = event_accumulator.Scalars(tag)
    if not scalars:
        return None
    return scalars[-1].value


def summarize_run(base_dir, run_name, keys):
    run_dir = os.path.join(base_dir, run_name)
    options_path = os.path.join(run_dir, "options.json")
    if not os.path.isfile(options_path):
        raise FileNotFoundError(options_path)
    with open(options_path, "r", encoding="utf-8") as f:
        options = json.load(f).get("training_options", {})

    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
    if not event_files:
        return options, {}
    event_file = max(event_files, key=os.path.getmtime)
    accumulator = EventAccumulator(event_file)
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    values = {}
    for prefix in ("Train", "Eval"):
        for key in keys:
            tag = f"{prefix}/{key}"
            if tag in tags:
                value = latest_scalar(accumulator, tag)
                if value is not None:
                    values[tag] = value
    return options, values


def fmt(value):
    if value is None:
        return ""
    return f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run names, e.g. OARM_50 OARM_51")
    parser.add_argument("--base-dir", default="OARM/saved")
    parser.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    args = parser.parse_args()

    rows = []
    for run_name in args.runs:
        options, values = summarize_run(args.base_dir, run_name, args.keys)
        rows.append(
            {
                "run": run_name,
                "config": os.path.basename(options.get("config", "")),
                "candidate": options.get("candidate_mode", ""),
                "margin": str(options.get("train_reaction_margin", "")),
                "ranking": str(options.get("train_margin_ranking", "")),
                "source": options.get("risk_label_source", ""),
                **values,
            }
        )

    columns = ["run", "config", "candidate", "margin", "ranking", "source"]
    for prefix in ("Train", "Eval"):
        for key in args.keys:
            col = f"{prefix}/{key}"
            if any(col in row for row in rows):
                columns.append(col)

    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt(value)
            widths[col] = max(widths[col], len(str(value)))

    print(" | ".join(col.ljust(widths[col]) for col in columns))
    print("-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt(value)
            cells.append(str(value).ljust(widths[col]))
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
