import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from OARM.dataset import OARMDataset
from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def project_points(depth, pos_w, rot_wb, risk_points_w):
    height, width = depth.shape[-2:]
    points_b = torch.matmul(
        rot_wb.transpose(0, 1),
        (risk_points_w - pos_w).unsqueeze(-1),
    ).squeeze(-1)
    x = points_b[:, 0].clamp(min=1e-4)
    y = points_b[:, 1]
    z = points_b[:, 2]
    yaw = torch.atan2(y, x)
    pitch = torch.atan2(z, torch.sqrt(x.square() + y.square()).clamp(min=1e-4))
    horizon_fov = math.radians(cfg["horizon_camera_fov"])
    vertical_fov = math.radians(cfg["vertical_camera_fov"])
    u = ((yaw / horizon_fov) + 0.5) * width - 0.5
    v = (0.5 - pitch / vertical_fov) * height - 0.5
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (x > 0)
    return u, v, in_image


def visualize(args):
    dataset = OARMDataset(
        mode=args.mode,
        dataset_root=args.dataset_root or None,
        use_privileged_risk_filter=args.use_privileged_risk_filter,
        risk_label_source=args.risk_label_source,
    )
    depth, pos, rot_wb, _obs_b, map_id, labels = dataset[args.index]
    depth_2d = depth.squeeze(0)
    pos_w = torch.as_tensor(pos, dtype=torch.float32)
    rot_wb = torch.as_tensor(rot_wb, dtype=torch.float32)
    risk_points_w = labels["risk_points_w"].float()
    risk_weight = labels["risk_weight"].float()
    valid = risk_weight > args.min_weight
    u, v, in_image = project_points(depth_2d, pos_w, rot_wb, risk_points_w)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(depth_2d.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    if bool(valid.any()):
        visible_valid = valid & in_image
        if bool(visible_valid.any()):
            weights = risk_weight[visible_valid].numpy()
            ax.scatter(
                u[visible_valid].numpy(),
                v[visible_valid].numpy(),
                c=weights,
                cmap="magma",
                s=35 + 90 * weights,
                edgecolors="cyan",
                linewidths=0.8,
            )
        off_image = valid & ~in_image
        if bool(off_image.any()):
            print(f"{int(off_image.sum())} valid GT risk points are outside the current depth image.")
    ax.set_title(
        f"idx={args.index} map={int(map_id)} source={args.risk_label_source} "
        f"valid={int(valid.sum())}/{risk_weight.numel()}"
    )
    ax.set_axis_off()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Wrote {args.output}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="")
    p.add_argument("--mode", choices=["train", "valid"], default="valid")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--risk-label-source", choices=["proxy", "proxy_esdf", "gt_pointcloud"], default="gt_pointcloud")
    p.add_argument("--use-privileged-risk-filter", action="store_true")
    p.add_argument("--min-weight", type=float, default=1e-6)
    p.add_argument("--output", default="OARM/results/gt_risk_points.png")
    p.add_argument("--dpi", type=int, default=160)
    return p


if __name__ == "__main__":
    visualize(parser().parse_args())
