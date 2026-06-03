#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Command-line pipeline for the beam-twist POD + neural latent ROM example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from newton.examples.mpm.rom.beam_twist_pod_nn import (
    fit_pod,
    generate_controlled_rollout,
    generate_rollout,
    read_json,
    render_comparison,
    rollout_latent_linear,
    rollout_latent_nn,
    train_linear_latent,
    train_mlp,
    write_json,
    write_pipeline_report,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate one teacher rollout.")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--run-name", type=str, required=True)
    gen.add_argument("--frames", type=int, default=60)
    gen.add_argument("--device", type=str, default="cpu")
    gen.add_argument("--grid-type", choices=["dense", "sparse", "fixed"], default="dense")
    gen.add_argument("--young-modulus", type=float, default=5.0e6)
    gen.add_argument("--damping", type=float, default=0.001)
    gen.add_argument("--twist-speed-scale", type=float, default=1.0)
    gen.add_argument("--voxel-size", type=float, default=0.25)
    gen.add_argument("--fps", type=float, default=240.0)
    gen.add_argument("--solver", type=str, default="cr")

    gen_controlled = sub.add_parser("generate-controlled", help="Generate one controlled beam rollout.")
    gen_controlled.add_argument("--output-dir", type=Path, required=True)
    gen_controlled.add_argument("--run-name", type=str, required=True)
    gen_controlled.add_argument("--frames", type=int, default=96)
    gen_controlled.add_argument("--device", type=str, default="cpu")
    gen_controlled.add_argument("--grid-type", choices=["dense", "sparse", "fixed"], default="dense")
    gen_controlled.add_argument("--young-modulus", type=float, default=5.0e6)
    gen_controlled.add_argument("--damping", type=float, default=0.001)
    gen_controlled.add_argument("--twist-speed-scale", type=float, default=1.0)
    gen_controlled.add_argument("--lateral-amp", type=float, default=0.0)
    gen_controlled.add_argument("--vertical-amp", type=float, default=0.0)
    gen_controlled.add_argument("--axial-amp", type=float, default=0.0)
    gen_controlled.add_argument("--lateral-freq-hz", type=float, default=0.7)
    gen_controlled.add_argument("--vertical-freq-hz", type=float, default=0.9)
    gen_controlled.add_argument("--axial-freq-hz", type=float, default=0.5)
    gen_controlled.add_argument("--lateral-phase", type=float, default=0.0)
    gen_controlled.add_argument("--vertical-phase", type=float, default=0.0)
    gen_controlled.add_argument("--axial-phase", type=float, default=0.0)
    gen_controlled.add_argument("--voxel-size", type=float, default=0.75)
    gen_controlled.add_argument("--fps", type=float, default=240.0)
    gen_controlled.add_argument("--solver", type=str, default="cr")

    pod = sub.add_parser("fit-pod", help="Fit a POD basis from teacher rollouts.")
    pod.add_argument("--output-dir", type=Path, required=True)
    pod.add_argument("--run-name", type=str, default="pod_rank16")
    pod.add_argument("--rank", type=int, default=16)
    pod.add_argument("rollouts", type=Path, nargs="+")

    train = sub.add_parser("train-nn", help="Train latent MLP dynamics.")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--run-name", type=str, default="pod_nn_latent")
    train.add_argument("--pod-dir", type=Path, required=True)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--epochs", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1.0e-3)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("rollouts", type=Path, nargs="+")

    linear = sub.add_parser("train-linear", help="Train a linear latent dynamics baseline.")
    linear.add_argument("--output-dir", type=Path, required=True)
    linear.add_argument("--run-name", type=str, default="pod_linear_latent")
    linear.add_argument("--pod-dir", type=Path, required=True)
    linear.add_argument("--ridge", type=float, default=1.0e-6)
    linear.add_argument("rollouts", type=Path, nargs="+")

    rollout = sub.add_parser("rollout", help="Run autoregressive held-out POD+NN rollout.")
    rollout.add_argument("--output-dir", type=Path, required=True)
    rollout.add_argument("--run-name", type=str, default="heldout_pod_nn_rollout")
    rollout.add_argument("--pod-dir", type=Path, required=True)
    rollout.add_argument("--model-dir", type=Path, required=True)
    rollout.add_argument("rollout", type=Path)

    rollout_linear = sub.add_parser("rollout-linear", help="Run autoregressive held-out POD+linear rollout.")
    rollout_linear.add_argument("--output-dir", type=Path, required=True)
    rollout_linear.add_argument("--run-name", type=str, default="heldout_pod_linear_rollout")
    rollout_linear.add_argument("--pod-dir", type=Path, required=True)
    rollout_linear.add_argument("--model-dir", type=Path, required=True)
    rollout_linear.add_argument("rollout", type=Path)

    render = sub.add_parser("render", help="Render comparison GIF/MP4 using ffmpeg.")
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--name", type=str, default="pod_nn_compare")
    render.add_argument("--fps", type=int, default=12)
    render.add_argument("--max-points", type=int, default=9000)
    render.add_argument("--prediction-dir", type=Path, required=True)
    render.add_argument("rollout", type=Path)

    report = sub.add_parser("report", help="Write a Markdown report for one pipeline run.")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--teacher-val", type=Path, required=True)
    report.add_argument("--pod-dir", type=Path, required=True)
    report.add_argument("--nn-model-dir", type=Path, required=True)
    report.add_argument("--nn-rollout-dir", type=Path, required=True)
    report.add_argument("--render-dir", type=Path)
    report.add_argument("--linear-model-dir", type=Path)
    report.add_argument("--linear-rollout-dir", type=Path)
    report.add_argument("teacher_train", type=Path, nargs="+")

    smoke = sub.add_parser("run-smoke", help="Run a tiny end-to-end smoke pipeline.")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device", type=str, default="cpu")
    smoke.add_argument("--grid-type", choices=["dense", "sparse", "fixed"], default="dense")
    smoke.add_argument("--frames", type=int, default=12)
    smoke.add_argument("--rank", type=int, default=8)
    smoke.add_argument("--epochs", type=int, default=200)
    smoke.add_argument("--voxel-size", type=float, default=0.5)

    dataset = sub.add_parser("run-controlled-dataset", help="Generate/train/evaluate a richer controlled beam dataset.")
    dataset.add_argument("--output-dir", type=Path, required=True)
    dataset.add_argument("--device", type=str, default="cpu")
    dataset.add_argument("--grid-type", choices=["dense", "sparse", "fixed"], default="dense")
    dataset.add_argument("--frames", type=int, default=96)
    dataset.add_argument("--rank", type=int, default=16)
    dataset.add_argument("--epochs", type=int, default=1500)
    dataset.add_argument("--voxel-size", type=float, default=0.75)
    dataset.add_argument("--hidden-dim", type=int, default=96)

    return p


def main() -> None:
    args = parser().parse_args()
    if args.cmd == "generate":
        print(
            generate_rollout(
                args.output_dir,
                run_name=args.run_name,
                frames=args.frames,
                device=args.device,
                grid_type=args.grid_type,
                young_modulus=args.young_modulus,
                damping=args.damping,
                twist_speed_scale=args.twist_speed_scale,
                voxel_size=args.voxel_size,
                fps=args.fps,
                solver=args.solver,
            )
        )
    elif args.cmd == "generate-controlled":
        print(
            generate_controlled_rollout(
                args.output_dir,
                run_name=args.run_name,
                frames=args.frames,
                device=args.device,
                grid_type=args.grid_type,
                young_modulus=args.young_modulus,
                damping=args.damping,
                twist_speed_scale=args.twist_speed_scale,
                lateral_amp=args.lateral_amp,
                vertical_amp=args.vertical_amp,
                axial_amp=args.axial_amp,
                lateral_freq_hz=args.lateral_freq_hz,
                vertical_freq_hz=args.vertical_freq_hz,
                axial_freq_hz=args.axial_freq_hz,
                lateral_phase=args.lateral_phase,
                vertical_phase=args.vertical_phase,
                axial_phase=args.axial_phase,
                voxel_size=args.voxel_size,
                fps=args.fps,
                solver=args.solver,
            )
        )
    elif args.cmd == "fit-pod":
        print(fit_pod(args.output_dir, args.rollouts, rank=args.rank, run_name=args.run_name))
    elif args.cmd == "train-nn":
        print(
            train_mlp(
                args.output_dir,
                args.rollouts,
                args.pod_dir,
                run_name=args.run_name,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            )
        )
    elif args.cmd == "train-linear":
        print(
            train_linear_latent(
                args.output_dir,
                args.rollouts,
                args.pod_dir,
                run_name=args.run_name,
                ridge=args.ridge,
            )
        )
    elif args.cmd == "rollout":
        print(rollout_latent_nn(args.output_dir, args.rollout, args.pod_dir, args.model_dir, run_name=args.run_name))
    elif args.cmd == "rollout-linear":
        print(
            rollout_latent_linear(
                args.output_dir,
                args.rollout,
                args.pod_dir,
                args.model_dir,
                run_name=args.run_name,
            )
        )
    elif args.cmd == "render":
        print(
            render_comparison(
                args.output_dir,
                args.rollout,
                args.prediction_dir,
                name=args.name,
                fps=args.fps,
                max_points=args.max_points,
            )
        )
    elif args.cmd == "report":
        print(
            write_pipeline_report(
                args.output,
                teacher_train=args.teacher_train,
                teacher_val=args.teacher_val,
                pod_dir=args.pod_dir,
                nn_model_dir=args.nn_model_dir,
                nn_rollout_dir=args.nn_rollout_dir,
                render_dir=args.render_dir,
                linear_model_dir=args.linear_model_dir,
                linear_rollout_dir=args.linear_rollout_dir,
            )
        )
    elif args.cmd == "run-smoke":
        root = args.output_dir
        train0 = generate_rollout(
            root / "teacher_rollouts",
            run_name="train_twist_0p9",
            frames=args.frames,
            device=args.device,
            grid_type=args.grid_type,
            twist_speed_scale=0.9,
            voxel_size=args.voxel_size,
        )
        train1 = generate_rollout(
            root / "teacher_rollouts",
            run_name="train_twist_1p1",
            frames=args.frames,
            device=args.device,
            grid_type=args.grid_type,
            twist_speed_scale=1.1,
            voxel_size=args.voxel_size,
        )
        val = generate_rollout(
            root / "teacher_rollouts",
            run_name="val_twist_1p0",
            frames=args.frames,
            device=args.device,
            grid_type=args.grid_type,
            twist_speed_scale=1.0,
            voxel_size=args.voxel_size,
        )
        pod = fit_pod(root / "pod_models", [train0, train1], rank=args.rank, run_name=f"pod_rank{args.rank}")
        linear_model = train_linear_latent(root / "latent_models", [train0, train1], pod)
        linear_pred = rollout_latent_linear(root / "reduced_rollouts", val, pod, linear_model)
        model = train_mlp(root / "latent_models", [train0, train1], pod, epochs=args.epochs)
        pred = rollout_latent_nn(root / "reduced_rollouts", val, pod, model)
        render_dir = render_comparison(root / "animations", val, pred, name="pod_nn_smoke_compare")
        report = write_pipeline_report(
            root / "beam_twist_pod_nn_report.md",
            teacher_train=[train0, train1],
            teacher_val=val,
            pod_dir=pod,
            nn_model_dir=model,
            nn_rollout_dir=pred,
            render_dir=render_dir,
            linear_model_dir=linear_model,
            linear_rollout_dir=linear_pred,
        )
        write_json(
            root / "smoke_summary.json",
            {
                "teacher_train": [str(train0), str(train1)],
                "teacher_val": str(val),
                "pod_model": str(pod),
                "linear_model": str(linear_model),
                "linear_rollout": str(linear_pred),
                "latent_model": str(model),
                "rollout": str(pred),
                "render": str(render_dir),
                "report": str(report),
            },
        )
        print(root / "smoke_summary.json")
    elif args.cmd == "run-controlled-dataset":
        run_controlled_dataset(args)


def controlled_dataset_specs() -> list[dict[str, Any]]:
    return [
        # Training cases: cover mostly-single-mode and combined controls.
        {
            "split": "train",
            "name": "train_twist_nominal",
            "twist": 1.00,
            "lat": 0.00,
            "vert": 0.00,
            "ax": 0.00,
            "lf": 0.70,
            "vf": 0.90,
            "af": 0.50,
            "lp": 0.0,
            "vp": 0.0,
            "ap": 0.0,
        },
        {
            "split": "train",
            "name": "train_vertical_low",
            "twist": 0.90,
            "lat": 0.00,
            "vert": 0.10,
            "ax": 0.00,
            "lf": 0.75,
            "vf": 0.65,
            "af": 0.50,
            "lp": 0.2,
            "vp": 0.0,
            "ap": 0.0,
        },
        {
            "split": "train",
            "name": "train_vertical_high",
            "twist": 1.10,
            "lat": 0.00,
            "vert": 0.18,
            "ax": 0.00,
            "lf": 0.65,
            "vf": 0.95,
            "af": 0.50,
            "lp": 0.0,
            "vp": 0.7,
            "ap": 0.0,
        },
        {
            "split": "train",
            "name": "train_lateral_low",
            "twist": 0.95,
            "lat": 0.10,
            "vert": 0.00,
            "ax": 0.00,
            "lf": 0.80,
            "vf": 0.90,
            "af": 0.50,
            "lp": 0.4,
            "vp": 0.0,
            "ap": 0.0,
        },
        {
            "split": "train",
            "name": "train_lateral_high",
            "twist": 1.05,
            "lat": 0.18,
            "vert": 0.00,
            "ax": 0.00,
            "lf": 1.05,
            "vf": 0.90,
            "af": 0.50,
            "lp": 0.9,
            "vp": 0.0,
            "ap": 0.0,
        },
        {
            "split": "train",
            "name": "train_axial_inout",
            "twist": 1.00,
            "lat": 0.00,
            "vert": 0.00,
            "ax": 0.08,
            "lf": 0.70,
            "vf": 0.90,
            "af": 0.85,
            "lp": 0.0,
            "vp": 0.0,
            "ap": 0.3,
        },
        {
            "split": "train",
            "name": "train_twist_vertical_axial",
            "twist": 0.92,
            "lat": 0.00,
            "vert": 0.14,
            "ax": 0.05,
            "lf": 0.70,
            "vf": 0.75,
            "af": 0.60,
            "lp": 0.0,
            "vp": 1.1,
            "ap": 0.5,
        },
        {
            "split": "train",
            "name": "train_twist_lateral_axial",
            "twist": 1.08,
            "lat": 0.14,
            "vert": 0.00,
            "ax": 0.05,
            "lf": 0.95,
            "vf": 0.90,
            "af": 0.70,
            "lp": 1.2,
            "vp": 0.0,
            "ap": 0.7,
        },
        {
            "split": "train",
            "name": "train_lateral_vertical_a",
            "twist": 0.98,
            "lat": 0.10,
            "vert": 0.12,
            "ax": 0.03,
            "lf": 0.70,
            "vf": 1.00,
            "af": 0.55,
            "lp": 0.6,
            "vp": 1.4,
            "ap": 0.2,
        },
        {
            "split": "train",
            "name": "train_lateral_vertical_b",
            "twist": 1.02,
            "lat": 0.16,
            "vert": 0.08,
            "ax": 0.06,
            "lf": 1.10,
            "vf": 0.80,
            "af": 0.95,
            "lp": 1.6,
            "vp": 0.3,
            "ap": 1.1,
        },
        {
            "split": "train",
            "name": "train_combo_soft",
            "twist": 0.88,
            "lat": 0.08,
            "vert": 0.16,
            "ax": 0.04,
            "lf": 0.55,
            "vf": 0.70,
            "af": 0.50,
            "lp": 2.0,
            "vp": 0.8,
            "ap": 0.6,
        },
        {
            "split": "train",
            "name": "train_combo_fast",
            "twist": 1.12,
            "lat": 0.12,
            "vert": 0.14,
            "ax": 0.07,
            "lf": 1.20,
            "vf": 1.15,
            "af": 1.00,
            "lp": 0.1,
            "vp": 1.7,
            "ap": 1.4,
        },
        # Validation interpolation.
        {
            "split": "val",
            "name": "val_interp_vertical_lateral",
            "twist": 1.00,
            "lat": 0.12,
            "vert": 0.10,
            "ax": 0.04,
            "lf": 0.85,
            "vf": 0.85,
            "af": 0.65,
            "lp": 0.8,
            "vp": 1.0,
            "ap": 0.4,
        },
        {
            "split": "val",
            "name": "val_interp_lateral_axial",
            "twist": 0.96,
            "lat": 0.15,
            "vert": 0.04,
            "ax": 0.06,
            "lf": 0.90,
            "vf": 0.75,
            "af": 0.80,
            "lp": 1.1,
            "vp": 0.2,
            "ap": 0.9,
        },
        {
            "split": "val",
            "name": "val_interp_vertical_axial",
            "twist": 1.06,
            "lat": 0.04,
            "vert": 0.15,
            "ax": 0.05,
            "lf": 0.75,
            "vf": 1.05,
            "af": 0.85,
            "lp": 0.5,
            "vp": 1.3,
            "ap": 1.0,
        },
        {
            "split": "val",
            "name": "val_interp_combo",
            "twist": 1.03,
            "lat": 0.11,
            "vert": 0.13,
            "ax": 0.05,
            "lf": 1.00,
            "vf": 0.95,
            "af": 0.75,
            "lp": 1.5,
            "vp": 0.6,
            "ap": 0.2,
        },
        # Extrapolation: outside one or more training ranges.
        {
            "split": "extra",
            "name": "extra_high_motion",
            "twist": 1.20,
            "lat": 0.22,
            "vert": 0.20,
            "ax": 0.09,
            "lf": 1.25,
            "vf": 1.20,
            "af": 1.05,
            "lp": 0.7,
            "vp": 1.5,
            "ap": 0.9,
        },
        {
            "split": "extra",
            "name": "extra_slow_large_vertical",
            "twist": 0.80,
            "lat": 0.06,
            "vert": 0.24,
            "ax": 0.08,
            "lf": 0.50,
            "vf": 0.60,
            "af": 0.45,
            "lp": 1.4,
            "vp": 0.4,
            "ap": 1.2,
        },
        {
            "split": "extra",
            "name": "extra_fast_lateral_axial",
            "twist": 1.18,
            "lat": 0.24,
            "vert": 0.06,
            "ax": 0.10,
            "lf": 1.35,
            "vf": 1.10,
            "af": 1.20,
            "lp": 0.3,
            "vp": 0.9,
            "ap": 1.6,
        },
    ]


def run_controlled_dataset(args: argparse.Namespace) -> None:
    root = args.output_dir
    specs = controlled_dataset_specs()
    generated: dict[str, list[Path]] = {"train": [], "val": [], "extra": []}
    spec_rows = []
    for spec in specs:
        rollout = generate_controlled_rollout(
            root / "teacher_rollouts",
            run_name=spec["name"],
            frames=args.frames,
            device=args.device,
            grid_type=args.grid_type,
            twist_speed_scale=spec["twist"],
            lateral_amp=spec["lat"],
            vertical_amp=spec["vert"],
            axial_amp=spec["ax"],
            lateral_freq_hz=spec["lf"],
            vertical_freq_hz=spec["vf"],
            axial_freq_hz=spec["af"],
            lateral_phase=spec["lp"],
            vertical_phase=spec["vp"],
            axial_phase=spec["ap"],
            voxel_size=args.voxel_size,
        )
        generated[spec["split"]].append(rollout)
        spec_rows.append({**spec, "path": str(rollout)})

    pod = fit_pod(root / "pod_models", generated["train"], rank=args.rank, run_name=f"pod_rank{args.rank}")
    linear_model = train_linear_latent(root / "latent_models", generated["train"], pod)
    nn_model = train_mlp(
        root / "latent_models",
        generated["train"],
        pod,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )

    eval_rows = []
    for split in ("val", "extra"):
        for rollout in generated[split]:
            linear_pred = rollout_latent_linear(
                root / "reduced_rollouts" / split,
                rollout,
                pod,
                linear_model,
                run_name=f"{rollout.name}_pod_linear",
            )
            nn_pred = rollout_latent_nn(
                root / "reduced_rollouts" / split,
                rollout,
                pod,
                nn_model,
                run_name=f"{rollout.name}_pod_nn",
            )
            eval_rows.append(
                {
                    "split": split,
                    "rollout": str(rollout),
                    "linear_rollout": str(linear_pred),
                    "nn_rollout": str(nn_pred),
                    "linear_report": read_json(linear_pred / "rollout_report.json"),
                    "nn_report": read_json(nn_pred / "rollout_report.json"),
                }
            )

    summary = {
        "teacher_rollouts": spec_rows,
        "train_rollouts": [str(path) for path in generated["train"]],
        "val_rollouts": [str(path) for path in generated["val"]],
        "extra_rollouts": [str(path) for path in generated["extra"]],
        "pod_model": str(pod),
        "linear_model": str(linear_model),
        "nn_model": str(nn_model),
        "evaluations": eval_rows,
    }
    write_json(root / "controlled_dataset_summary.json", summary)
    write_controlled_dataset_report(root / "controlled_dataset_report.md", summary)
    print(root / "controlled_dataset_summary.json")


def write_controlled_dataset_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Controlled Beam POD + NN Dataset Report",
        "",
        "## Dataset",
        "",
        f"- Train rollouts: {len(summary['train_rollouts'])}",
        f"- Validation rollouts: {len(summary['val_rollouts'])}",
        f"- Extrapolation rollouts: {len(summary['extra_rollouts'])}",
        f"- POD model: {summary['pod_model']}",
        f"- NN model: {summary['nn_model']}",
        "",
        "The driven beam end uses combined twist, lateral translation, vertical translation, and axial in/out translation.",
        "",
        "## Cases",
        "",
        "| Split | Name | Twist | Lateral amp | Vertical amp | Axial amp |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["teacher_rollouts"]:
        lines.append(
            f"| {row['split']} | {row['name']} | {row['twist']:.2f} | {row['lat']:.3f} | {row['vert']:.3f} | {row['ax']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Held-Out Metrics",
            "",
            "| Split | Case | Model | Position RMSE (m) | Max Error (m) | Velocity Proxy RMSE (m/s) | Online FPS |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["evaluations"]:
        case = Path(row["rollout"]).name
        for model_name, report_key in (("POD+NN", "nn_report"), ("POD+linear", "linear_report")):
            report = row[report_key]
            q = report["quality"]
            t = report["timing"]
            lines.append(
                f"| {row['split']} | {case} | {model_name} | {q['position_rmse_m']:.6g} | "
                f"{q['position_max_l2_m']:.6g} | {q['velocity_proxy_rmse_m_per_s']:.6g} | {t['online_fps']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
