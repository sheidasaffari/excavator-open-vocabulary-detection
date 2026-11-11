#!/usr/bin/env python3
"""Run MM-Grounding-DINO common inference and dump MMDet predictions."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


RUNS = Path("/home/ssaffari/Data/foundational/journal_study/runs_phase1/mm_grounding_dino_t")
PRETRAINED = Path("/home/ssaffari/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth")
TEST_TOOL = Path("/home/ssaffari/miniconda3/envs/mmdet/lib/python3.10/site-packages/mmdet/.mim/tools/test.py")
PYTHON = Path("/home/ssaffari/miniconda3/envs/mmdet/bin/python")
MODELS = {
    "base": (RUNS / "n2_fusion_seed0/locked_config.py", PRETRAINED),
    "n0_fusion": (RUNS / "n0_fusion_seed0/locked_config.py", next((RUNS / "n0_fusion_seed0").glob("best_*.pth"))),
    "n1_fusion": (RUNS / "n1_fusion_seed0/locked_config.py", next((RUNS / "n1_fusion_seed0").glob("best_*.pth"))),
    "n2_head": (RUNS / "n2_head_seed0/locked_config.py", next((RUNS / "n2_head_seed0").glob("best_*.pth"))),
    "n2_fusion": (RUNS / "n2_fusion_seed0/locked_config.py", next((RUNS / "n2_fusion_seed0").glob("best_*.pth"))),
    "n2_visionfusion": (RUNS / "n2_visionfusion_seed0/locked_config.py", next((RUNS / "n2_visionfusion_seed0").glob("best_*.pth"))),
    "n2_full": (RUNS / "n2_full_seed0/locked_config.py", next((RUNS / "n2_full_seed0").glob("best_*.pth"))),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=tuple(MODELS), required=True)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        config, checkpoint = MODELS[name]
        output = args.output_root / f"{name}.pkl"
        work = args.output_root / f"work_{name}"
        if output.exists():
            print(f"Skip existing {output}", flush=True)
            continue
        cmd = [
            str(PYTHON), str(TEST_TOOL), str(config), str(checkpoint),
            "--work-dir", str(work), "--out", str(output),
            "--cfg-options",
            f"test_dataloader.dataset.ann_file={args.eval_json.resolve()}",
            "test_dataloader.dataset.data_root=",
            "test_dataloader.dataset.data_prefix.img=",
            f"test_evaluator.ann_file={args.eval_json.resolve()}",
        ]
        print("RUN", name, flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
