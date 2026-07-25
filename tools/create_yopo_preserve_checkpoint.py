import argparse
import json
import os

import torch

from OARM.policy.oarm_network import OARMNetwork
from OARM.utils.checkpoint import make_oarm_checkpoint


def parser():
    p = argparse.ArgumentParser(description="Create an OARM A0 checkpoint from an official YOPO checkpoint.")
    p.add_argument("--yopo-checkpoint", required=True, help="official YOPO .pth checkpoint")
    p.add_argument("--output", required=True, help="output OARM checkpoint path")
    p.add_argument("--stage", default="a0_yopo_preserve")
    return p


def main(args):
    if not os.path.isfile(args.yopo_checkpoint):
        raise FileNotFoundError(f"YOPO checkpoint not found: {args.yopo_checkpoint}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = OARMNetwork(candidate_mode="yopo_preserve", backbone_mode="yopo_original").to(device)
    state_dict = torch.load(args.yopo_checkpoint, map_location=device, weights_only=True)
    policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)
    checkpoint = make_oarm_checkpoint(
        policy.state_dict(),
        candidate_mode="yopo_preserve",
        backbone_mode="yopo_original",
        training_options={
            "stage": args.stage,
            "candidate_mode": "yopo_preserve",
            "backbone_mode": "yopo_original",
            "source_yopo_checkpoint": os.path.abspath(args.yopo_checkpoint),
            "preserves_yopo_endpoint": True,
            "preserves_yopo_score": True,
            "preserves_yopo_traj_time": True,
            "oarm_aux_heads_affect_selection": False,
        },
        enable_yield_candidates=False,
        deployed_yaw_mode="goal",
        risk_label_source="proxy",
    )
    torch.save(checkpoint, args.output)
    print(
        json.dumps(
            {
                "output": os.path.abspath(args.output),
                "source_yopo_checkpoint": os.path.abspath(args.yopo_checkpoint),
                "candidate_mode": "yopo_preserve",
                "backbone_mode": "yopo_original",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(parser().parse_args())
