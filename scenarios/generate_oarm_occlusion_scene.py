"""Generate deterministic OARM occlusion benchmark pointclouds.

The generated PLY can be used by the YOPO Simulator through
`random_map: false` and as the matched GT pointcloud for OARM postprocess.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def frange(start: float, stop: float, step: float):
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        yield start + i * step


def add_surface_box(points: set[tuple[int, int, int]], bounds, res: float) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xs = list(frange(xmin, xmax, res))
    ys = list(frange(ymin, ymax, res))
    zs = list(frange(zmin, zmax, res))
    for x in xs:
        for y in ys:
            for z in zs:
                on_surface = (
                    abs(x - xmin) < 1e-6
                    or abs(x - xmax) < 1e-6
                    or abs(y - ymin) < 1e-6
                    or abs(y - ymax) < 1e-6
                    or abs(z - zmin) < 1e-6
                    or abs(z - zmax) < 1e-6
                )
                if on_surface:
                    points.add((round(x / res), round(y / res), round(z / res)))


def add_ground(points: set[tuple[int, int, int]], xlim, ylim, res: float, ground_res: float) -> None:
    for x in frange(xlim[0], xlim[1], ground_res):
        for y in frange(ylim[0], ylim[1], ground_res):
            points.add((round(x / res), round(y / res), 0))


def blind_gate_s0(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 4.5

    # A visible gate directly in front of the vehicle. Both sides are open, but
    # the left-side opening hides a later trap behind the gate.
    boxes = [
        (7.8, 10.0, -1.35, 1.35, 0.0, h),
        (12.5, 18.5, 2.4, 6.8, 0.0, h),
        (17.5, 18.7, -7.0, -2.8, 0.0, h),
        (25.0, 26.2, -3.0, 2.2, 0.0, h),
        (32.0, 33.5, 2.0, 6.8, 0.0, h),
        (39.0, 40.0, -6.5, -2.0, 0.0, h),
        (19.0, 21.0, -0.7, 0.7, 0.0, 3.4),
        (44.0, 45.2, 3.0, 6.5, 0.0, h),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-10.0, 10.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_s0",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "negative-y side of the first gate, then weave through mid-field occluders",
        "risk_design": "positive-y side after the first gate contains a hidden stopper occluded at approach",
        "boxes": boxes,
    }
    return points, meta


def blind_gate_s1(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 4.5
    boxes = [
        (8.0, 10.2, -1.2, 1.2, 0.0, h),
        (12.0, 18.0, -6.6, -2.5, 0.0, h),
        (17.5, 18.8, 2.8, 7.0, 0.0, h),
        (24.0, 25.2, -2.2, 3.0, 0.0, h),
        (31.5, 33.0, -6.8, -2.0, 0.0, h),
        (38.0, 39.2, 2.0, 6.5, 0.0, h),
        (18.8, 20.8, -0.7, 0.7, 0.0, 3.4),
        (44.0, 45.2, -6.5, -3.0, 0.0, h),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-10.0, 10.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_s1",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "positive-y side of the first gate, mirrored from s0",
        "risk_design": "negative-y side after the first gate contains a hidden stopper occluded at approach",
        "boxes": boxes,
    }
    return points, meta

def blind_gate_v2_s0(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 4.2
    boxes = [
        (8.0, 10.0, -0.8, 5.2, 0.0, h),
        (13.0, 14.4, -1.2, 1.1, 0.0, 3.4),
        (16.0, 18.0, -5.2, 0.8, 0.0, h),
        (21.0, 22.4, -1.1, 1.2, 0.0, 3.4),
        (24.0, 26.0, -0.8, 5.2, 0.0, h),
        (29.0, 30.4, -1.2, 1.1, 0.0, 3.4),
        (32.0, 34.0, -5.2, 0.8, 0.0, h),
        (37.0, 38.2, -1.0, 1.0, 0.0, 3.2),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-9.0, 9.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_v2_s0",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "slalom through alternating lower and upper openings, then straight to goal",
        "risk_design": "short center stoppers are hidden behind partial walls during approach",
        "boxes": boxes,
    }
    return points, meta


def blind_gate_v2_s1(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 4.2
    boxes = [
        (8.0, 10.0, -5.2, 0.8, 0.0, h),
        (13.0, 14.4, -1.1, 1.2, 0.0, 3.4),
        (16.0, 18.0, -0.8, 5.2, 0.0, h),
        (21.0, 22.4, -1.2, 1.1, 0.0, 3.4),
        (24.0, 26.0, -5.2, 0.8, 0.0, h),
        (29.0, 30.4, -1.1, 1.2, 0.0, 3.4),
        (32.0, 34.0, -0.8, 5.2, 0.0, h),
        (37.0, 38.2, -1.0, 1.0, 0.0, 3.2),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-9.0, 9.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_v2_s1",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "mirrored slalom through alternating openings, then straight to goal",
        "risk_design": "mirrored center stoppers hidden behind partial walls during approach",
        "boxes": boxes,
    }
    return points, meta
def blind_gate_v3_s0(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 4.0
    boxes = [
        (7.0, 8.6, -0.4, 4.8, 0.0, h),
        (10.8, 12.0, -1.0, 1.0, 0.0, 3.2),
        (13.8, 15.4, -4.8, 0.4, 0.0, h),
        (17.6, 18.8, -1.0, 1.0, 0.0, 3.2),
        (21.0, 22.6, -0.4, 4.8, 0.0, h),
        (24.8, 26.0, -1.0, 1.0, 0.0, 3.2),
        (28.2, 29.8, -4.8, 0.4, 0.0, h),
        (32.0, 33.2, -1.0, 1.0, 0.0, 3.2),
        (35.0, 36.4, -0.4, 4.2, 0.0, h),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-8.0, 8.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_v3_s0",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "stay near the alternating open side while passing close to center hidden stoppers",
        "risk_design": "center stoppers are close to the nominal route and partly hidden by upstream walls",
        "boxes": boxes,
    }
    return points, meta


def blind_gate_v3_s1(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 4.0
    boxes = [
        (7.0, 8.6, -4.8, 0.4, 0.0, h),
        (10.8, 12.0, -1.0, 1.0, 0.0, 3.2),
        (13.8, 15.4, -0.4, 4.8, 0.0, h),
        (17.6, 18.8, -1.0, 1.0, 0.0, 3.2),
        (21.0, 22.6, -4.8, 0.4, 0.0, h),
        (24.8, 26.0, -1.0, 1.0, 0.0, 3.2),
        (28.2, 29.8, -0.4, 4.8, 0.0, h),
        (32.0, 33.2, -1.0, 1.0, 0.0, 3.2),
        (35.0, 36.4, -4.2, 0.4, 0.0, h),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-8.0, 8.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_v3_s1",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "mirrored close-pass slalom with center hidden stoppers",
        "risk_design": "mirrored center stoppers close to the nominal route",
        "boxes": boxes,
    }
    return points, meta
def blind_gate_v4_s0(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 3.8
    boxes = [
        # Corridor rails prevent the policy from escaping around all occluders.
        (4.0, 42.0, -6.2, -5.7, 0.0, h),
        (4.0, 42.0, 5.7, 6.2, 0.0, h),
        # Alternating partial gates in the corridor.
        (8.0, 9.4, -5.7, 1.2, 0.0, h),
        (12.2, 13.2, -1.0, 1.0, 0.0, 3.1),
        (15.5, 16.9, -1.2, 5.7, 0.0, h),
        (19.7, 20.7, -1.0, 1.0, 0.0, 3.1),
        (23.0, 24.4, -5.7, 1.2, 0.0, h),
        (27.2, 28.2, -1.0, 1.0, 0.0, 3.1),
        (30.5, 31.9, -1.2, 5.7, 0.0, h),
        (34.7, 35.7, -1.0, 1.0, 0.0, 3.1),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-8.0, 8.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_v4_s0",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "follow the bounded slalom corridor and pass center stoppers with clearance",
        "risk_design": "center stoppers and alternating gates create repeated hidden near-route risks",
        "boxes": boxes,
    }
    return points, meta


def blind_gate_v4_s1(res: float) -> tuple[set[tuple[int, int, int]], dict]:
    points: set[tuple[int, int, int]] = set()
    h = 3.8
    boxes = [
        (4.0, 42.0, -6.2, -5.7, 0.0, h),
        (4.0, 42.0, 5.7, 6.2, 0.0, h),
        (8.0, 9.4, -1.2, 5.7, 0.0, h),
        (12.2, 13.2, -1.0, 1.0, 0.0, 3.1),
        (15.5, 16.9, -5.7, 1.2, 0.0, h),
        (19.7, 20.7, -1.0, 1.0, 0.0, 3.1),
        (23.0, 24.4, -1.2, 5.7, 0.0, h),
        (27.2, 28.2, -1.0, 1.0, 0.0, 3.1),
        (30.5, 31.9, -5.7, 1.2, 0.0, h),
        (34.7, 35.7, -1.0, 1.0, 0.0, 3.1),
    ]
    for box in boxes:
        add_surface_box(points, box, res)
    add_ground(points, (-6.0, 58.0), (-8.0, 8.0), res, ground_res=0.25)
    meta = {
        "scene": "blind_gate_v4_s1",
        "goal": [50.0, 0.0, 2.0],
        "safe_route_hint": "mirrored bounded slalom corridor",
        "risk_design": "mirrored repeated hidden near-route risks",
        "boxes": boxes,
    }
    return points, meta

SCENES = {
    "blind_gate_s0": blind_gate_s0,
    "blind_gate_s1": blind_gate_s1,
    "blind_gate_v2_s0": blind_gate_v2_s0,
    "blind_gate_v2_s1": blind_gate_v2_s1,
    "blind_gate_v3_s0": blind_gate_v3_s0,
    "blind_gate_v3_s1": blind_gate_v3_s1,
    "blind_gate_v4_s0": blind_gate_v4_s0,
    "blind_gate_v4_s1": blind_gate_v4_s1,
}


def write_ply(path: Path, points: set[tuple[int, int, int]], res: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(points)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(ordered)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for ix, iy, iz in ordered:
            f.write(f"{ix * res:.3f} {iy * res:.3f} {iz * res:.3f}\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=sorted(SCENES), default="blind_gate_s0")
    p.add_argument("--output-ply", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--resolution", type=float, default=0.1)
    return p


def main() -> None:
    args = parser().parse_args()
    output_ply = args.output_ply or Path("OARM/scenarios/generated") / f"oarm_{args.scene}.ply"
    points, meta = SCENES[args.scene](args.resolution)
    write_ply(output_ply, points, args.resolution)

    dataset_pointcloud = None
    if args.dataset_dir is not None:
        args.dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_pointcloud = args.dataset_dir / "pointcloud-0.ply"
        shutil.copy2(output_ply, dataset_pointcloud)

    meta_path = output_ply.with_suffix(".json")
    meta = {
        **meta,
        "resolution": args.resolution,
        "point_count": len(points),
        "output_ply": str(output_ply),
        "dataset_pointcloud": str(dataset_pointcloud) if dataset_pointcloud else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

