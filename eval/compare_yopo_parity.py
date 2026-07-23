import argparse
import json
import os

import numpy as np


DEFAULT_THRESHOLDS = {
    "normalized_obs": 1e-6,
    "primitive_obs": 1e-6,
    "raw_endstate": 1e-5,
    "raw_score": 1e-5,
    "selected_endstate_b": 1e-4,
    "selected_endstate_w": 1e-4,
    "decoded_endstate_b_all": 1e-4,
    "decoded_endstate_w_all": 1e-4,
    "trajectory_time": 1e-7,
    "control_pos_w": 1e-4,
    "control_vel_w": 1e-4,
    "control_acc_w": 1e-3,
    "control_yaw": 1e-4,
    "control_yaw_dot": 1e-4,
    "control_flag": 0.0,
}

EXACT_KEYS = ("selected_id", "selected_lattice_id")


def scalarize(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def compare_array(key, ref, cand, threshold):
    if key not in ref.files or key not in cand.files:
        return {
            "key": key,
            "status": "missing",
            "threshold": threshold,
            "reference_present": key in ref.files,
            "candidate_present": key in cand.files,
        }
    ref_value = np.asarray(ref[key])
    cand_value = np.asarray(cand[key])
    if ref_value.shape != cand_value.shape:
        return {
            "key": key,
            "status": "shape_mismatch",
            "threshold": threshold,
            "reference_shape": list(ref_value.shape),
            "candidate_shape": list(cand_value.shape),
        }
    if key in EXACT_KEYS:
        equal = bool(np.array_equal(ref_value, cand_value))
        return {
            "key": key,
            "status": "pass" if equal else "fail",
            "reference": scalarize(ref_value),
            "candidate": scalarize(cand_value),
            "max_abs_error": 0.0 if equal else None,
            "threshold": 0.0,
        }
    diff = np.asarray(ref_value, dtype=np.float64) - np.asarray(cand_value, dtype=np.float64)
    max_abs_error = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    return {
        "key": key,
        "status": "pass" if max_abs_error <= threshold else "fail",
        "max_abs_error": max_abs_error,
        "threshold": float(threshold),
        "reference_shape": list(ref_value.shape),
        "candidate_shape": list(cand_value.shape),
    }


def summarize(results):
    failures = [r for r in results if r["status"] != "pass"]
    return {
        "passed": len(failures) == 0,
        "num_checks": len(results),
        "num_failures": len(failures),
        "failures": failures,
        "results": results,
    }


def parser():
    p = argparse.ArgumentParser(description="Compare two YOPO parity NPZ dumps.")
    p.add_argument("--reference", required=True, help="reference NPZ, e.g. clean YOPO or trusted exact-adapter dump")
    p.add_argument("--candidate", required=True, help="candidate NPZ to validate")
    p.add_argument("--output", default="", help="optional JSON report path")
    p.add_argument("--fail-on-mismatch", action="store_true")
    p.add_argument(
        "--keys",
        nargs="*",
        default=list(DEFAULT_THRESHOLDS.keys()) + list(EXACT_KEYS),
        help="keys to compare; defaults to the parity contract keys",
    )
    return p


def main(args):
    with np.load(args.reference, allow_pickle=False) as ref, np.load(args.candidate, allow_pickle=False) as cand:
        results = []
        for key in args.keys:
            threshold = DEFAULT_THRESHOLDS.get(key, 0.0)
            results.append(compare_array(key, ref, cand, threshold))
    report = summarize(results)
    report["reference"] = os.path.abspath(args.reference)
    report["candidate"] = os.path.abspath(args.candidate)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.fail_on_mismatch and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(parser().parse_args())
