"""Create a non-destructive channel-importance report before structured pruning.

The report ranks BatchNorm channels by absolute gamma magnitude. It is an importance
screening aid, not pruning itself: dependency-aware structural pruning is required
before any channels can be physically removed from a YOLOv5 graph.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.utils.patches import torch_load


LAYER_ROLES = {
    range(0, 4): "stem_p2",
    range(4, 6): "p3_backbone_protect",
    range(6, 8): "p4_backbone_candidate",
    range(8, 11): "p5_backbone_candidate",
    range(11, 15): "topdown_p4_candidate",
    range(15, 19): "p3_head_protect",
    range(19, 22): "bottomup_p4_candidate",
    range(22, 25): "bottomup_p5_candidate",
    range(25, 26): "detect_protect",
}


def layer_role(module_name):
    """Return the YOLO macro-layer index, role, and pruning protection flag."""
    match = re.match(r"model\.(\d+)(?:\.|$)", module_name)
    if not match:
        return "", "other", False
    layer_index = int(match.group(1))
    for indices, role in LAYER_ROLES.items():
        if layer_index in indices:
            return layer_index, role, role.endswith("protect")
    return layer_index, "other", False


def quantile(values, q):
    """Return a scalar quantile compatible with PyTorch checkpoints on CPU."""
    return float(torch.quantile(values, q).item())


def analyze(weights, output, align):
    """Load a checkpoint and save one channel-importance summary row per BatchNorm layer."""
    checkpoint = torch_load(weights, map_location="cpu")
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        gamma = module.weight.detach().abs().cpu()
        layer_index, role, protected = layer_role(module_name)
        rows.append(
            {
                "module": module_name,
                "macro_layer": layer_index,
                "role": role,
                "protect_from_initial_pruning": protected,
                "channels": gamma.numel(),
                "channels_aligned_to_8": gamma.numel() % align == 0,
                "mean_abs_gamma": float(gamma.mean().item()),
                "median_abs_gamma": quantile(gamma, 0.5),
                "p10_abs_gamma": quantile(gamma, 0.1),
                "min_abs_gamma": float(gamma.min().item()),
            }
        )

    fieldnames = list(rows[0]) if rows else []
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    protected = sum(row["protect_from_initial_pruning"] for row in rows)
    candidates = len(rows) - protected
    print(f"Saved {len(rows)} BatchNorm summaries to {output}")
    print(f"Initial pruning policy: protect {protected} P3/Detect summaries, screen {candidates} P4/P5 summaries.")
    print(f"Channel counts must remain multiples of {align} for FPGA-friendly pruning.")


def parse_opt():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="trained unfused .pt checkpoint")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/paper/pruning_analysis/channel_importance.csv"), help="CSV output"
    )
    parser.add_argument("--align", type=int, default=8, help="required retained-channel alignment")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_opt()
    analyze(options.weights, options.output, options.align)
