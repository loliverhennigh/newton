# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""POD + neural latent dynamics utilities for the MPM beam-twist example."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.mpm.example_mpm_beam_twist import Example

SCHEMA_VERSION = "newton_mpm_beam_twist_pod_nn_rom_v1"


@wp.kernel
def apply_twist_translate(
    indices: wp.array[int],
    rel_pos: wp.array[wp.vec3],
    out_pos: wp.array[wp.vec3],
    out_vel: wp.array[wp.vec3],
    out_vel_grad: wp.array[wp.mat33],
    center: wp.vec3,
    angle: float,
    speed: float,
    translation: wp.vec3,
    translation_vel: wp.vec3,
):
    tid = wp.tid()
    idx = indices[tid]
    r = rel_pos[tid]

    s = wp.sin(angle)
    c = wp.cos(angle)
    ry = r[1] * c - r[2] * s
    rz = r[1] * s + r[2] * c

    out_pos[idx] = center + translation + wp.vec3(r[0], ry, rz)
    out_vel[idx] = translation_vel + wp.vec3(0.0, -rz * speed, ry * speed)
    out_vel_grad[idx] = wp.skew(wp.vec3(speed, 0.0, 0.0))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _beam_parser_args(
    *,
    device: str,
    grid_type: str,
    frames: int,
    young_modulus: float,
    damping: float,
    voxel_size: float,
    fps: float,
    solver: str,
) -> argparse.Namespace:
    parser = Example.create_parser()
    args = parser.parse_args(
        [
            "--viewer",
            "null",
            "--device",
            device,
            "--num-frames",
            str(frames),
            "--young-modulus",
            str(young_modulus),
            "--damping",
            str(damping),
            "--voxel-size",
            str(voxel_size),
            "--fps",
            str(fps),
            "--solver",
            solver,
            "--quiet",
        ]
    )
    args.grid_type = grid_type
    return args


def generate_rollout(
    output_dir: Path,
    *,
    run_name: str,
    frames: int,
    device: str = "cpu",
    grid_type: str = "dense",
    young_modulus: float = 5.0e6,
    damping: float = 0.001,
    twist_speed_scale: float = 1.0,
    voxel_size: float = 0.25,
    fps: float = 240.0,
    solver: str = "cr",
) -> Path:
    """Generate one beam-twist teacher rollout and save arrays plus a manifest."""

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args = _beam_parser_args(
        device=device,
        grid_type=grid_type,
        frames=frames,
        young_modulus=young_modulus,
        damping=damping,
        voxel_size=voxel_size,
        fps=fps,
        solver=solver,
    )
    wp.init()
    wp.set_device(args.device)
    viewer = newton.viewer.ViewerNull(num_frames=frames)
    example = Example(viewer, args)
    example.twist_speed *= float(twist_speed_scale)

    particle_count = int(example.model.particle_count)
    particle_q = np.empty((frames + 1, particle_count, 3), dtype=np.float32)
    particle_qd = np.empty_like(particle_q)
    time_values = np.empty((frames + 1,), dtype=np.float64)
    step_times_ms = []

    for frame in range(frames + 1):
        particle_q[frame] = example.state_0.particle_q.numpy()
        particle_qd[frame] = example.state_0.particle_qd.numpy()
        time_values[frame] = float(example.sim_time)
        if frame < frames:
            step_start = time.perf_counter()
            example.step()
            wp.synchronize()
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)

    rest_position = particle_q[0].copy()
    twist_indices = example.twist_indices.numpy().astype(np.int32)
    collider_points = particle_q[:, twist_indices].copy()
    collider_velocities = particle_qd[:, twist_indices].copy()
    np.save(run_dir / "particle_q.npy", particle_q)
    np.save(run_dir / "particle_qd.npy", particle_qd)
    np.save(run_dir / "rest_position.npy", rest_position)
    np.save(run_dir / "time.npy", time_values)
    np.save(run_dir / "twist_indices.npy", twist_indices)
    np.save(run_dir / "collider_points.npy", collider_points)
    np.save(run_dir / "collider_velocities.npy", collider_velocities)

    manifest = {
        "schema": f"{SCHEMA_VERSION}.teacher_rollout",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "particle_count": particle_count,
        "frames": int(frames),
        "config": {
            "device": device,
            "grid_type": grid_type,
            "young_modulus": float(young_modulus),
            "damping": float(damping),
            "twist_speed_scale": float(twist_speed_scale),
            "base_twist_speed": float(example.twist_speed / float(twist_speed_scale)),
            "twist_speed": float(example.twist_speed),
            "twist_frames": int(example.twist_frames),
            "voxel_size": float(voxel_size),
            "fps": float(fps),
            "frame_dt": float(example.frame_dt),
            "sim_dt": float(example.sim_dt),
            "solver": solver,
        },
        "timing": {
            "step_total_ms": float(np.sum(step_times_ms)),
            "step_mean_ms_per_frame": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
            "step_min_ms_per_frame": float(np.min(step_times_ms)) if step_times_ms else 0.0,
            "step_max_ms_per_frame": float(np.max(step_times_ms)) if step_times_ms else 0.0,
            "step_fps": float(1000.0 / max(np.mean(step_times_ms), 1.0e-12)) if step_times_ms else 0.0,
        },
        "files": {
            "particle_q": "particle_q.npy",
            "particle_qd": "particle_qd.npy",
            "rest_position": "rest_position.npy",
            "time": "time.npy",
            "twist_indices": "twist_indices.npy",
            "collider_points": "collider_points.npy",
            "collider_velocities": "collider_velocities.npy",
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _oscillatory_displacement(t: float, amplitude: float, frequency_hz: float, phase: float) -> tuple[float, float]:
    omega = 2.0 * math.pi * float(frequency_hz)
    displacement = float(amplitude) * (math.sin(omega * t + float(phase)) - math.sin(float(phase)))
    velocity = float(amplitude) * omega * math.cos(omega * t + float(phase))
    return displacement, velocity


def controlled_motion(config: dict[str, Any], t: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    twist_speed = float(config["twist_speed"])
    angle = twist_speed * t
    tx, vx = _oscillatory_displacement(t, config["axial_amp"], config["axial_freq_hz"], config["axial_phase"])
    ty, vy = _oscillatory_displacement(t, config["lateral_amp"], config["lateral_freq_hz"], config["lateral_phase"])
    tz, vz = _oscillatory_displacement(t, config["vertical_amp"], config["vertical_freq_hz"], config["vertical_phase"])
    return angle, twist_speed, np.array([tx, ty, tz], dtype=np.float32), np.array([vx, vy, vz], dtype=np.float32)


def generate_controlled_rollout(
    output_dir: Path,
    *,
    run_name: str,
    frames: int,
    device: str = "cpu",
    grid_type: str = "dense",
    young_modulus: float = 5.0e6,
    damping: float = 0.001,
    twist_speed_scale: float = 1.0,
    lateral_amp: float = 0.0,
    vertical_amp: float = 0.0,
    axial_amp: float = 0.0,
    lateral_freq_hz: float = 0.7,
    vertical_freq_hz: float = 0.9,
    axial_freq_hz: float = 0.5,
    lateral_phase: float = 0.0,
    vertical_phase: float = 0.0,
    axial_phase: float = 0.0,
    voxel_size: float = 0.75,
    fps: float = 240.0,
    solver: str = "cr",
) -> Path:
    """Generate a teacher rollout with twist plus end translations.

    The right end of the beam remains a kinematic boundary. Compared with the stock
    beam-twist example, this drives that boundary with a combined axial/lateral/
    vertical sinusoidal translation and twist rotation.
    """

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args = _beam_parser_args(
        device=device,
        grid_type=grid_type,
        frames=frames,
        young_modulus=young_modulus,
        damping=damping,
        voxel_size=voxel_size,
        fps=fps,
        solver=solver,
    )
    wp.init()
    wp.set_device(args.device)
    viewer = newton.viewer.ViewerNull(num_frames=frames)
    example = Example(viewer, args)
    base_twist_speed = float(example.twist_speed)
    example.twist_speed = base_twist_speed * float(twist_speed_scale)

    motion_config = {
        "control_mode": "twist_translate",
        "twist_speed_scale": float(twist_speed_scale),
        "base_twist_speed": base_twist_speed,
        "twist_speed": float(example.twist_speed),
        "lateral_amp": float(lateral_amp),
        "vertical_amp": float(vertical_amp),
        "axial_amp": float(axial_amp),
        "lateral_freq_hz": float(lateral_freq_hz),
        "vertical_freq_hz": float(vertical_freq_hz),
        "axial_freq_hz": float(axial_freq_hz),
        "lateral_phase": float(lateral_phase),
        "vertical_phase": float(vertical_phase),
        "axial_phase": float(axial_phase),
    }

    particle_count = int(example.model.particle_count)
    particle_q = np.empty((frames + 1, particle_count, 3), dtype=np.float32)
    particle_qd = np.empty_like(particle_q)
    time_values = np.empty((frames + 1,), dtype=np.float64)
    step_times_ms = []

    def apply_boundary() -> None:
        angle, speed, translation, translation_vel = controlled_motion(motion_config, float(example.sim_time))
        wp.launch(
            kernel=apply_twist_translate,
            dim=example.twist_indices.shape[0],
            inputs=[
                example.twist_indices,
                example.twist_rel_pos,
                example.state_0.particle_q,
                example.state_0.particle_qd,
                example.state_0.mpm.particle_qd_grad,
                wp.vec3(float(example.twist_center[0]), float(example.twist_center[1]), float(example.twist_center[2])),
                float(angle),
                float(speed),
                wp.vec3(float(translation[0]), float(translation[1]), float(translation[2])),
                wp.vec3(float(translation_vel[0]), float(translation_vel[1]), float(translation_vel[2])),
            ],
            device=example.model.device,
        )

    for frame in range(frames + 1):
        apply_boundary()
        wp.synchronize()
        particle_q[frame] = example.state_0.particle_q.numpy()
        particle_qd[frame] = example.state_0.particle_qd.numpy()
        time_values[frame] = float(example.sim_time)
        if frame < frames:
            step_start = time.perf_counter()
            example.simulate()
            example.sim_time += example.frame_dt
            wp.synchronize()
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)

    rest_position = particle_q[0].copy()
    twist_indices = example.twist_indices.numpy().astype(np.int32)
    collider_points = particle_q[:, twist_indices].copy()
    collider_velocities = particle_qd[:, twist_indices].copy()
    np.save(run_dir / "particle_q.npy", particle_q)
    np.save(run_dir / "particle_qd.npy", particle_qd)
    np.save(run_dir / "rest_position.npy", rest_position)
    np.save(run_dir / "time.npy", time_values)
    np.save(run_dir / "twist_indices.npy", twist_indices)
    np.save(run_dir / "collider_points.npy", collider_points)
    np.save(run_dir / "collider_velocities.npy", collider_velocities)

    manifest = {
        "schema": f"{SCHEMA_VERSION}.teacher_rollout",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "particle_count": particle_count,
        "frames": int(frames),
        "config": {
            "device": device,
            "grid_type": grid_type,
            "young_modulus": float(young_modulus),
            "damping": float(damping),
            "twist_frames": int(example.twist_frames),
            "voxel_size": float(voxel_size),
            "fps": float(fps),
            "frame_dt": float(example.frame_dt),
            "sim_dt": float(example.sim_dt),
            "solver": solver,
            **motion_config,
        },
        "timing": {
            "step_total_ms": float(np.sum(step_times_ms)),
            "step_mean_ms_per_frame": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
            "step_min_ms_per_frame": float(np.min(step_times_ms)) if step_times_ms else 0.0,
            "step_max_ms_per_frame": float(np.max(step_times_ms)) if step_times_ms else 0.0,
            "step_fps": float(1000.0 / max(np.mean(step_times_ms), 1.0e-12)) if step_times_ms else 0.0,
        },
        "files": {
            "particle_q": "particle_q.npy",
            "particle_qd": "particle_qd.npy",
            "rest_position": "rest_position.npy",
            "time": "time.npy",
            "twist_indices": "twist_indices.npy",
            "collider_points": "collider_points.npy",
            "collider_velocities": "collider_velocities.npy",
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def load_rollout(path: Path) -> dict[str, Any]:
    manifest = read_json(path / "manifest.json")
    files = manifest["files"]
    rollout = {
        "path": path,
        "manifest": manifest,
        "particle_q": np.load(path / files["particle_q"]).astype(np.float32),
        "particle_qd": np.load(path / files["particle_qd"]).astype(np.float32),
        "rest_position": np.load(path / files["rest_position"]).astype(np.float32),
        "time": np.load(path / files["time"]).astype(np.float64),
        "twist_indices": np.load(path / files["twist_indices"]).astype(np.int32),
    }
    for key in ("collider_points", "collider_normals", "collider_velocities"):
        if key in files:
            rollout[key] = np.load(path / files[key]).astype(np.float32)
    return rollout


def fit_pod(output_dir: Path, rollout_dirs: list[Path], *, rank: int, run_name: str = "pod_rank16") -> Path:
    if rank < 1:
        raise ValueError("POD rank must be positive")
    rollouts = [load_rollout(path) for path in rollout_dirs]
    rest = rollouts[0]["rest_position"]
    particle_count = int(rest.shape[0])
    for rollout in rollouts:
        if rollout["rest_position"].shape != rest.shape:
            raise ValueError("All rollouts must use the same particle layout")

    snapshots = []
    for rollout in rollouts:
        snapshots.append(
            (rollout["particle_q"] - rest.reshape(1, particle_count, 3)).reshape(rollout["particle_q"].shape[0], -1)
        )
    snapshot_matrix = np.concatenate(snapshots, axis=0).astype(np.float64)
    mean_delta = snapshot_matrix.mean(axis=0).astype(np.float32)
    centered = snapshot_matrix - mean_delta.reshape(1, -1)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    actual_rank = min(rank, vt.shape[0])
    basis = vt[:actual_rank].astype(np.float32)

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "mean_delta.npy", mean_delta)
    np.save(run_dir / "basis.npy", basis)
    np.save(run_dir / "singular_values.npy", singular_values.astype(np.float32))
    np.save(run_dir / "rest_position.npy", rest.astype(np.float32))

    coeff_files = []
    for rollout in rollouts:
        coeffs = project_to_pod(rollout["particle_q"], rest, mean_delta, basis)
        name = f"{rollout['path'].name}_coeffs.npy"
        np.save(run_dir / name, coeffs)
        coeff_files.append({"rollout": str(rollout["path"]), "path": name})

    manifest = {
        "schema": f"{SCHEMA_VERSION}.pod_model",
        "generated_at": datetime.now(UTC).isoformat(),
        "rank": int(actual_rank),
        "requested_rank": int(rank),
        "particle_count": particle_count,
        "rollouts": [str(path) for path in rollout_dirs],
        "decoder": "q = rest_position + mean_delta + basis.T @ z",
        "files": {
            "mean_delta": "mean_delta.npy",
            "basis": "basis.npy",
            "singular_values": "singular_values.npy",
            "rest_position": "rest_position.npy",
            "coefficients": coeff_files,
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def load_pod(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "manifest": read_json(path / "manifest.json"),
        "mean_delta": np.load(path / "mean_delta.npy").astype(np.float32),
        "basis": np.load(path / "basis.npy").astype(np.float32),
        "rest_position": np.load(path / "rest_position.npy").astype(np.float32),
    }


def _normalize(v: np.ndarray, *, axis: int = -1, eps: float = 1.0e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def _fallback_collider_samples(rollout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Use the driven beam-end particles as a sampled kinematic collider."""

    indices = rollout["twist_indices"]
    points = rollout["particle_q"][:, indices]
    velocities = rollout["particle_qd"][:, indices]
    normals = rollout.get("collider_normals")
    return points.astype(np.float32), normals, velocities.astype(np.float32)


def particle_collider_fields(
    rollout: dict[str, Any],
    *,
    frames: int | None = None,
    surface_radius: float = 0.075,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Compute per-particle nearest-collider context fields.

    The returned field has seven channels per particle:
    signed distance, nearest surface normal xyz, and relative collider velocity xyz.
    If a rollout supplies sampled collider normals, signed distance is the normal
    projection against the nearest sample. Otherwise the nearest-sample radial
    direction is used and the distance is offset by ``surface_radius``.
    """

    q = rollout["particle_q"] if frames is None else rollout["particle_q"][:frames]
    qd = rollout["particle_qd"] if frames is None else rollout["particle_qd"][:frames]
    points = rollout.get("collider_points")
    normals = rollout.get("collider_normals")
    velocities = rollout.get("collider_velocities")
    if points is None or velocities is None:
        points, normals, velocities = _fallback_collider_samples(rollout)
    points = points[: q.shape[0]].astype(np.float32)
    velocities = velocities[: q.shape[0]].astype(np.float32)
    if normals is not None:
        normals = _normalize(normals[: q.shape[0]].astype(np.float32))

    fields = np.empty((q.shape[0], q.shape[1], 7), dtype=np.float32)
    for frame in range(q.shape[0]):
        for start in range(0, q.shape[1], chunk_size):
            end = min(start + chunk_size, q.shape[1])
            delta = q[frame, start:end, None, :] - points[frame, None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", delta, delta)
            nearest = np.argmin(dist2, axis=1)
            nearest_delta = delta[np.arange(end - start), nearest]
            nearest_vel = velocities[frame, nearest]
            if normals is None:
                nearest_normal = _normalize(nearest_delta)
                signed_distance = np.sqrt(dist2[np.arange(end - start), nearest]) - float(surface_radius)
            else:
                nearest_normal = normals[frame, nearest]
                signed_distance = np.einsum("ij,ij->i", nearest_delta, nearest_normal)
            relative_velocity = qd[frame, start:end] - nearest_vel
            fields[frame, start:end, 0] = signed_distance
            fields[frame, start:end, 1:4] = nearest_normal
            fields[frame, start:end, 4:7] = relative_velocity
    return fields


def fit_collider_condition_pod(
    output_dir: Path,
    rollout_dirs: list[Path],
    *,
    rank: int,
    run_name: str = "collider_condition_pod_rank8",
) -> Path:
    """Fit a POD basis for geometry-derived collider/contact conditioning fields."""

    if rank < 1:
        raise ValueError("Conditioning POD rank must be positive")
    rollouts = [load_rollout(path) for path in rollout_dirs]
    field_shape = particle_collider_fields(rollouts[0], frames=1).shape[1:]
    snapshots = []
    for rollout in rollouts:
        fields = particle_collider_fields(rollout)
        if fields.shape[1:] != field_shape:
            raise ValueError("All rollouts must use the same particle/collider field layout")
        snapshots.append(fields.reshape(fields.shape[0], -1))
    snapshot_matrix = np.concatenate(snapshots, axis=0).astype(np.float64)
    mean_field = snapshot_matrix.mean(axis=0).astype(np.float32)
    centered = snapshot_matrix - mean_field.reshape(1, -1)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    actual_rank = min(rank, vt.shape[0])

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "mean_field.npy", mean_field)
    np.save(run_dir / "basis.npy", vt[:actual_rank].astype(np.float32))
    np.save(run_dir / "singular_values.npy", singular_values.astype(np.float32))
    manifest = {
        "schema": f"{SCHEMA_VERSION}.collider_condition_pod",
        "generated_at": datetime.now(UTC).isoformat(),
        "rank": int(actual_rank),
        "requested_rank": int(rank),
        "field_shape": list(field_shape),
        "channels": [
            "signed_distance_to_nearest_collider_sample",
            "nearest_normal_x",
            "nearest_normal_y",
            "nearest_normal_z",
            "relative_velocity_x",
            "relative_velocity_y",
            "relative_velocity_z",
        ],
        "rollouts": [str(path) for path in rollout_dirs],
        "decoder": "field = mean_field + condition_basis.T @ c",
        "files": {
            "mean_field": "mean_field.npy",
            "basis": "basis.npy",
            "singular_values": "singular_values.npy",
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def load_collider_condition_pod(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "manifest": read_json(path / "manifest.json"),
        "mean_field": np.load(path / "mean_field.npy").astype(np.float32),
        "basis": np.load(path / "basis.npy").astype(np.float32),
    }


def project_collider_condition_fields(fields: np.ndarray, condition_pod: dict[str, Any]) -> np.ndarray:
    flat = fields.reshape(fields.shape[0], -1)
    centered = flat - condition_pod["mean_field"].reshape(1, -1)
    return (centered @ condition_pod["basis"].T).astype(np.float32)


def project_to_pod(q: np.ndarray, rest: np.ndarray, mean_delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    delta = (q - rest.reshape(1, *rest.shape)).reshape(q.shape[0], -1) - mean_delta.reshape(1, -1)
    return (delta @ basis.T).astype(np.float32)


def decode_pod(z: np.ndarray, rest: np.ndarray, mean_delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    flat = mean_delta.reshape(1, -1) + z.astype(np.float32) @ basis.astype(np.float32)
    return rest.reshape(1, *rest.shape) + flat.reshape(z.shape[0], *rest.shape)


def control_features(
    rollout: dict[str, Any],
    frames: int | None = None,
    *,
    condition_pod_dir: Path | None = None,
) -> np.ndarray:
    time_values = rollout["time"] if frames is None else rollout["time"][:frames]
    config = rollout["manifest"]["config"]
    ym = np.full_like(time_values, float(config["young_modulus"]) / 5.0e6)
    damping = np.full_like(time_values, float(config.get("damping", 0.001)) / 0.001)
    if condition_pod_dir is not None:
        condition_pod = load_collider_condition_pod(condition_pod_dir)
        fields = particle_collider_fields(rollout, frames=frames)
        collider_coeffs = project_collider_condition_fields(fields, condition_pod)
        frame_dt = np.full_like(time_values, float(config.get("frame_dt", 1.0 / 240.0)) * 240.0)
        return np.concatenate([np.stack([ym, damping, frame_dt], axis=1), collider_coeffs], axis=1).astype(np.float32)

    twist_speed = float(config["twist_speed"])
    angle = twist_speed * time_values
    rate = np.full_like(angle, twist_speed)
    scale = np.full_like(angle, float(config["twist_speed_scale"]))
    if config.get("control_mode") != "twist_translate":
        return np.stack([angle, np.sin(angle), np.cos(angle), rate, ym, damping, scale], axis=1).astype(np.float32)

    translations = []
    translation_velocities = []
    for t in time_values:
        _angle, _speed, translation, translation_vel = controlled_motion(config, float(t))
        translations.append(translation)
        translation_velocities.append(translation_vel)
    translation_arr = np.asarray(translations, dtype=np.float32)
    velocity_arr = np.asarray(translation_velocities, dtype=np.float32)

    constants = np.stack(
        [
            np.full_like(angle, float(config["lateral_amp"])),
            np.full_like(angle, float(config["vertical_amp"])),
            np.full_like(angle, float(config["axial_amp"])),
            np.full_like(angle, float(config["lateral_freq_hz"])),
            np.full_like(angle, float(config["vertical_freq_hz"])),
            np.full_like(angle, float(config["axial_freq_hz"])),
            np.full_like(angle, float(config["lateral_phase"])),
            np.full_like(angle, float(config["vertical_phase"])),
            np.full_like(angle, float(config["axial_phase"])),
        ],
        axis=1,
    )
    return np.concatenate(
        [
            np.stack([angle, np.sin(angle), np.cos(angle), rate, ym, damping, scale], axis=1),
            translation_arr,
            velocity_arr,
            constants,
        ],
        axis=1,
    ).astype(np.float32)


def latent_input_description(feature_count: int) -> str:
    """Describe the latent model input vector for manifests and reports."""

    fields = [
        "z_t",
        "z_t - z_{t-1}",
        "control_features",
    ]
    return f"concat({', '.join(fields)}), where control_features has {feature_count} columns"


def latent_training_data(
    rollout_dirs: list[Path],
    pod_dir: Path,
    *,
    condition_pod_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pod = load_pod(pod_dir)
    inputs = []
    targets = []
    feature_count = None
    for path in rollout_dirs:
        rollout = load_rollout(path)
        z = project_to_pod(rollout["particle_q"], pod["rest_position"], pod["mean_delta"], pod["basis"])
        controls = control_features(rollout, condition_pod_dir=condition_pod_dir)
        if feature_count is None:
            feature_count = controls.shape[1]
        elif controls.shape[1] != feature_count:
            raise ValueError("All training rollouts must use the same control feature layout")
        dz_prev = np.zeros_like(z[:-1])
        dz_prev[1:] = z[1:-1] - z[:-2]
        x = np.concatenate([z[:-1], dz_prev, controls[:-1]], axis=1)
        y = z[1:] - z[:-1]
        inputs.append(x.astype(np.float32))
        targets.append(y.astype(np.float32))
    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)


def latent_feature_count(rollout_dirs: list[Path], *, condition_pod_dir: Path | None = None) -> int:
    if not rollout_dirs:
        raise ValueError("At least one rollout is required")
    count = int(control_features(load_rollout(rollout_dirs[0]), frames=1, condition_pod_dir=condition_pod_dir).shape[1])
    for path in rollout_dirs[1:]:
        if int(control_features(load_rollout(path), frames=1, condition_pod_dir=condition_pod_dir).shape[1]) != count:
            raise ValueError("All training rollouts must use the same control feature layout")
    return count


@dataclass
class MLP:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w_skip: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.x_mean.reshape(1, -1)) / self.x_std.reshape(1, -1)
        h = np.tanh(xn @ self.w1 + self.b1.reshape(1, -1))
        yn = h @ self.w2 + xn @ self.w_skip + self.b2.reshape(1, -1)
        return yn * self.y_std.reshape(1, -1) + self.y_mean.reshape(1, -1)


@dataclass
class LinearLatentModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    weights: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.x_mean.reshape(1, -1)) / self.x_std.reshape(1, -1)
        xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=xn.dtype)], axis=1)
        yn = xa @ self.weights
        return yn * self.y_std.reshape(1, -1) + self.y_mean.reshape(1, -1)


def train_linear_latent(
    output_dir: Path,
    train_rollouts: list[Path],
    pod_dir: Path,
    *,
    run_name: str = "pod_linear_latent",
    ridge: float = 1.0e-6,
    condition_pod_dir: Path | None = None,
) -> Path:
    """Fit a ridge-regularized linear latent update baseline."""

    feature_count = latent_feature_count(train_rollouts, condition_pod_dir=condition_pod_dir)
    x, y = latent_training_data(train_rollouts, pod_dir, condition_pod_dir=condition_pod_dir)
    x_mean = x.mean(axis=0)
    x_std = np.maximum(x.std(axis=0), 1.0e-6)
    y_mean = y.mean(axis=0)
    y_std = np.maximum(y.std(axis=0), 1.0e-6)
    xn = (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    yn = (y - y_mean.reshape(1, -1)) / y_std.reshape(1, -1)
    xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=xn.dtype)], axis=1).astype(np.float64)
    lhs = xa.T @ xa
    lhs += float(ridge) * np.eye(lhs.shape[0], dtype=np.float64)
    lhs[-1, -1] -= float(ridge)
    rhs = xa.T @ yn.astype(np.float64)
    weights = np.linalg.solve(lhs, rhs).astype(np.float32)

    model = LinearLatentModel(x_mean, x_std, y_mean, y_std, weights)
    train_pred = model.predict(x)
    train_rmse = float(np.sqrt(np.mean(np.sum((train_pred - y) ** 2, axis=1))))

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "model.npz",
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        weights=weights,
    )
    manifest = {
        "schema": f"{SCHEMA_VERSION}.latent_linear",
        "generated_at": datetime.now(UTC).isoformat(),
        "pod_model": str(pod_dir),
        "condition_pod_model": str(condition_pod_dir) if condition_pod_dir else None,
        "conditioning_mode": "collider_pod" if condition_pod_dir else "scripted_control",
        "train_rollouts": [str(path) for path in train_rollouts],
        "input": latent_input_description(feature_count),
        "control_feature_count": feature_count,
        "target": "z_{t+1} - z_t",
        "ridge": float(ridge),
        "train_delta_z_rmse": train_rmse,
        "files": {"model": "model.npz"},
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def train_mlp(
    output_dir: Path,
    train_rollouts: list[Path],
    pod_dir: Path,
    *,
    run_name: str = "pod_nn_latent",
    hidden_dim: int = 64,
    epochs: int = 2000,
    batch_size: int = 256,
    lr: float = 1.0e-3,
    seed: int = 1234,
    condition_pod_dir: Path | None = None,
) -> Path:
    feature_count = latent_feature_count(train_rollouts, condition_pod_dir=condition_pod_dir)
    x, y = latent_training_data(train_rollouts, pod_dir, condition_pod_dir=condition_pod_dir)
    rng = np.random.default_rng(seed)
    x_mean = x.mean(axis=0)
    x_std = np.maximum(x.std(axis=0), 1.0e-6)
    y_mean = y.mean(axis=0)
    y_std = np.maximum(y.std(axis=0), 1.0e-6)
    xn = (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    yn = (y - y_mean.reshape(1, -1)) / y_std.reshape(1, -1)

    input_dim = xn.shape[1]
    output_dim = yn.shape[1]
    xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=xn.dtype)], axis=1).astype(np.float64)
    lhs = xa.T @ xa
    lhs += 1.0e-6 * np.eye(lhs.shape[0], dtype=np.float64)
    lhs[-1, -1] -= 1.0e-6
    rhs = xa.T @ yn.astype(np.float64)
    linear_init = np.linalg.solve(lhs, rhs).astype(np.float32)

    w1 = rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, output_dim), dtype=np.float32)
    b2 = linear_init[-1].copy()
    w_skip = linear_init[:-1].copy()

    # Adam state
    m = [np.zeros_like(arr) for arr in (w1, b1, w2, b2, w_skip)]
    v = [np.zeros_like(arr) for arr in (w1, b1, w2, b2, w_skip)]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    losses = []
    n = xn.shape[0]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            step += 1
            idx = order[start : start + batch_size]
            xb = xn[idx]
            yb = yn[idx]
            h_pre = xb @ w1 + b1.reshape(1, -1)
            h = np.tanh(h_pre)
            pred = h @ w2 + xb @ w_skip + b2.reshape(1, -1)
            err = (pred - yb).astype(np.float32)
            loss = float(np.mean(err * err))

            grad_pred = (2.0 / max(err.size, 1)) * err
            gw2 = h.T @ grad_pred
            gb2 = grad_pred.sum(axis=0)
            gw_skip = xb.T @ grad_pred
            gh = grad_pred @ w2.T
            gh_pre = gh * (1.0 - h * h)
            gw1 = xb.T @ gh_pre
            gb1 = gh_pre.sum(axis=0)
            grads = [
                gw1.astype(np.float32),
                gb1.astype(np.float32),
                gw2.astype(np.float32),
                gb2.astype(np.float32),
                gw_skip.astype(np.float32),
            ]
            params = [w1, b1, w2, b2, w_skip]
            for i, grad in enumerate(grads):
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                v[i] = beta2 * v[i] + (1.0 - beta2) * (grad * grad)
                mh = m[i] / (1.0 - beta1**step)
                vh = v[i] / (1.0 - beta2**step)
                params[i] -= lr * mh / (np.sqrt(vh) + eps)
        if epoch == 0 or (epoch + 1) % max(epochs // 20, 1) == 0:
            losses.append({"epoch": int(epoch + 1), "mse": loss})

    model = MLP(x_mean, x_std, y_mean, y_std, w1, b1, w2, b2, w_skip)
    train_pred = model.predict(x)
    train_rmse = float(np.sqrt(np.mean(np.sum((train_pred - y) ** 2, axis=1))))

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "model.npz",
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w_skip=w_skip,
    )
    manifest = {
        "schema": f"{SCHEMA_VERSION}.latent_mlp",
        "generated_at": datetime.now(UTC).isoformat(),
        "pod_model": str(pod_dir),
        "condition_pod_model": str(condition_pod_dir) if condition_pod_dir else None,
        "conditioning_mode": "collider_pod" if condition_pod_dir else "scripted_control",
        "train_rollouts": [str(path) for path in train_rollouts],
        "input": latent_input_description(feature_count),
        "control_feature_count": feature_count,
        "target": "z_{t+1} - z_t",
        "hidden_dim": int(hidden_dim),
        "architecture": "one_hidden_tanh_mlp_with_linear_skip",
        "initialization": "linear_skip_initialized_by_ridge_regression",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "train_delta_z_rmse": train_rmse,
        "losses": losses,
        "files": {"model": "model.npz"},
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def load_mlp(path: Path) -> MLP:
    arrays = np.load(path / "model.npz")
    w1 = arrays["w1"].astype(np.float32)
    w2 = arrays["w2"].astype(np.float32)
    if "w_skip" in arrays:
        w_skip = arrays["w_skip"].astype(np.float32)
    else:
        w_skip = np.zeros((w1.shape[0], w2.shape[1]), dtype=np.float32)
    return MLP(
        arrays["x_mean"].astype(np.float32),
        arrays["x_std"].astype(np.float32),
        arrays["y_mean"].astype(np.float32),
        arrays["y_std"].astype(np.float32),
        w1,
        arrays["b1"].astype(np.float32),
        w2,
        arrays["b2"].astype(np.float32),
        w_skip,
    )


def load_linear_latent(path: Path) -> LinearLatentModel:
    arrays = np.load(path / "model.npz")
    return LinearLatentModel(
        arrays["x_mean"].astype(np.float32),
        arrays["x_std"].astype(np.float32),
        arrays["y_mean"].astype(np.float32),
        arrays["y_std"].astype(np.float32),
        arrays["weights"].astype(np.float32),
    )


def rollout_latent_nn(
    output_dir: Path,
    rollout_dir: Path,
    pod_dir: Path,
    model_dir: Path,
    *,
    run_name: str = "heldout_pod_nn_rollout",
    condition_pod_dir: Path | None = None,
) -> Path:
    return _rollout_latent_model(
        output_dir,
        rollout_dir,
        pod_dir,
        load_mlp(model_dir),
        model_dir,
        run_name=run_name,
        model_kind="pod_nn",
        honesty="Autoregressive POD+NN latent rollout; uses only the held-out initial state and known conditioning features.",
        condition_pod_dir=condition_pod_dir,
    )


def rollout_latent_linear(
    output_dir: Path,
    rollout_dir: Path,
    pod_dir: Path,
    model_dir: Path,
    *,
    run_name: str = "heldout_pod_linear_rollout",
    condition_pod_dir: Path | None = None,
) -> Path:
    return _rollout_latent_model(
        output_dir,
        rollout_dir,
        pod_dir,
        load_linear_latent(model_dir),
        model_dir,
        run_name=run_name,
        model_kind="pod_linear",
        honesty="Autoregressive POD+linear latent rollout baseline; uses only the held-out initial state and known conditioning features.",
        condition_pod_dir=condition_pod_dir,
    )


def _rollout_latent_model(
    output_dir: Path,
    rollout_dir: Path,
    pod_dir: Path,
    model: MLP | LinearLatentModel,
    model_dir: Path,
    *,
    run_name: str,
    model_kind: str,
    honesty: str,
    condition_pod_dir: Path | None = None,
) -> Path:
    rollout = load_rollout(rollout_dir)
    pod = load_pod(pod_dir)
    model_manifest = read_json(model_dir / "manifest.json")
    if condition_pod_dir is None and model_manifest.get("condition_pod_model"):
        condition_pod_dir = Path(model_manifest["condition_pod_model"])
    target_q = rollout["particle_q"]
    target_z = project_to_pod(target_q, pod["rest_position"], pod["mean_delta"], pod["basis"])
    controls = control_features(rollout, condition_pod_dir=condition_pod_dir)
    z = np.empty_like(target_z)
    z[0] = target_z[0]
    dz_prev = np.zeros_like(z[0])
    start = time.perf_counter()
    for i in range(target_z.shape[0] - 1):
        x = np.concatenate([z[i], dz_prev, controls[i]], axis=0).reshape(1, -1).astype(np.float32)
        dz = model.predict(x).reshape(-1).astype(np.float32)
        z[i + 1] = z[i] + dz
        dz_prev = dz
    latent_ms = (time.perf_counter() - start) * 1000.0

    decode_start = time.perf_counter()
    pred_q = decode_pod(z, pod["rest_position"], pod["mean_delta"], pod["basis"]).astype(np.float32)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    metrics = evaluate_prediction(pred_q, rollout)

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "reduced_coordinates.npy", z)
    np.save(run_dir / "decoded_particle_q.npy", pred_q)
    timing = {
        "latent_update_total_ms": float(latent_ms),
        "decode_total_ms": float(decode_ms),
        "online_ms_per_frame": float((latent_ms + decode_ms) / max(target_z.shape[0] - 1, 1)),
        "online_fps": float(1000.0 * max(target_z.shape[0] - 1, 1) / max(latent_ms + decode_ms, 1.0e-12)),
    }
    teacher_timing = rollout["manifest"].get("timing", {})
    if teacher_timing:
        timing["teacher_step_mean_ms_per_frame"] = float(teacher_timing.get("step_mean_ms_per_frame", 0.0))
        timing["teacher_step_fps"] = float(teacher_timing.get("step_fps", 0.0))
        timing["speedup_vs_teacher_step_mean"] = float(
            timing["teacher_step_mean_ms_per_frame"] / max(timing["online_ms_per_frame"], 1.0e-12)
        )
    report = {
        "schema": f"{SCHEMA_VERSION}.rollout_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "rollout": str(rollout_dir),
        "pod_model": str(pod_dir),
        "latent_model": str(model_dir),
        "condition_pod_model": str(condition_pod_dir) if condition_pod_dir else None,
        "conditioning_mode": "collider_pod" if condition_pod_dir else "scripted_control",
        "model_kind": model_kind,
        "particle_count": int(pred_q.shape[1]),
        "frame_count": int(pred_q.shape[0]),
        "quality": metrics,
        "timing": timing,
        "files": {"reduced_coordinates": "reduced_coordinates.npy", "decoded_particle_q": "decoded_particle_q.npy"},
        "honesty": honesty,
    }
    write_json(run_dir / "rollout_report.json", report)
    write_json(run_dir / "manifest.json", {"schema": f"{SCHEMA_VERSION}.rollout_manifest", **report})
    return run_dir


def write_pipeline_report(
    output_path: Path,
    *,
    teacher_train: list[Path],
    teacher_val: Path,
    pod_dir: Path,
    nn_model_dir: Path,
    nn_rollout_dir: Path,
    render_dir: Path | None = None,
    linear_model_dir: Path | None = None,
    linear_rollout_dir: Path | None = None,
) -> Path:
    """Write a compact Markdown report for one beam-twist POD+NN run."""

    pod_manifest = read_json(pod_dir / "manifest.json")
    nn_manifest = read_json(nn_model_dir / "manifest.json")
    nn_report = read_json(nn_rollout_dir / "rollout_report.json")
    val_manifest = read_json(teacher_val / "manifest.json")
    linear_manifest = read_json(linear_model_dir / "manifest.json") if linear_model_dir else None
    linear_report = read_json(linear_rollout_dir / "rollout_report.json") if linear_rollout_dir else None
    config = val_manifest["config"]
    smoke_command = (
        "uv run python tools/mpm_beam_twist_pod_nn_rom.py run-smoke "
        f"--output-dir {output_path.parent} "
        f"--device {config['device']} --grid-type {config['grid_type']} "
        f"--frames {val_manifest['frames']} --rank {pod_manifest['rank']} "
        f"--epochs {nn_manifest['epochs']} --voxel-size {config['voxel_size']}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Beam-Twist POD + Neural Latent ROM Report",
        "",
        "## Summary",
        "",
        "This run uses Newton's beam-twist implicit-MPM example as the teacher, fits a POD basis on training rollouts, trains an MLP to predict latent POD updates, and evaluates an autonomous held-out rollout. The reduced model decodes full particle positions but does not call the Newton MPM solver during online rollout.",
        "",
        "## Commands",
        "",
        "The end-to-end smoke/benchmark command is:",
        "",
        "```bash",
        smoke_command,
        "```",
        "",
        "Individual pipeline commands are available through `generate`, `fit-pod`, `train-linear`, `train-nn`, `rollout-linear`, `rollout`, `render`, and `report` subcommands.",
        "",
        "## Dataset",
        "",
        f"- Training rollouts: {', '.join(str(path) for path in teacher_train)}",
        f"- Held-out rollout: {teacher_val}",
        f"- Particles: {val_manifest['particle_count']}",
        f"- Frames: {val_manifest['frames']}",
        f"- Teacher mean step time: {val_manifest.get('timing', {}).get('step_mean_ms_per_frame', 0.0):.4f} ms/frame",
        f"- Teacher FPS: {val_manifest.get('timing', {}).get('step_fps', 0.0):.2f}",
        "",
        "## POD Model",
        "",
        f"- POD model: {pod_dir}",
        f"- Rank: {pod_manifest['rank']}",
        f"- Decoder: `{pod_manifest['decoder']}`",
        "",
        "## Neural Latent Model",
        "",
        f"- Model: {nn_model_dir}",
        f"- Architecture: {nn_manifest.get('architecture', 'one_hidden_tanh_mlp')}",
        f"- Hidden dimension: {nn_manifest['hidden_dim']}",
        f"- Epochs: {nn_manifest['epochs']}",
        f"- Train latent update RMSE: {nn_manifest['train_delta_z_rmse']:.6g}",
        "",
        "## Held-Out Metrics",
        "",
        "| Model | Position RMSE (m) | Max Error (m) | Velocity Proxy RMSE (m/s) | Deformation Proxy RMSE | Online ms/frame | Online FPS | Speedup vs Teacher |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metric_row("POD+NN", nn_report),
    ]
    if linear_manifest and linear_report:
        lines.extend(
            [
                _metric_row("POD+linear", linear_report),
                "",
                "## Linear Baseline",
                "",
                f"- Model: {linear_model_dir}",
                f"- Ridge: {linear_manifest['ridge']}",
                f"- Train latent update RMSE: {linear_manifest['train_delta_z_rmse']:.6g}",
            ]
        )
    lines.extend(
        [
            "",
            "## Visual Artifacts",
            "",
            f"- Comparison render directory: {render_dir}"
            if render_dir
            else "- Comparison render directory: not generated",
            "",
            "## Limitations",
            "",
            "- This is a geometry-specific ROM for one beam particle layout and one narrow control family.",
            "- The online timing excludes any rendering or data transfer overhead and should be compared against the recorded teacher MPM step timing with that caveat.",
            "- The quality metrics are particle-position based proxies; a production version should add task-specific stress/energy checks.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def _metric_row(name: str, report: dict[str, Any]) -> str:
    q = report["quality"]
    t = report["timing"]
    return (
        f"| {name} | {q['position_rmse_m']:.6g} | {q['position_max_l2_m']:.6g} | "
        f"{q['velocity_proxy_rmse_m_per_s']:.6g} | {q['deformation_gradient_proxy_rmse']:.6g} | "
        f"{t['online_ms_per_frame']:.6g} | {t['online_fps']:.2f} | {t.get('speedup_vs_teacher_step_mean', 0.0):.2f}x |"
    )


def evaluate_prediction(pred_q: np.ndarray, rollout: dict[str, Any]) -> dict[str, Any]:
    target_q = rollout["particle_q"]
    err = pred_q - target_q
    l2 = np.linalg.norm(err, axis=-1)
    velocity = velocity_proxy(pred_q, rollout)
    deformation = deformation_proxy(pred_q, target_q, rollout["rest_position"])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(err * err, axis=-1)))),
        "position_max_l2_m": float(np.max(l2)),
        "position_mean_l2_m": float(np.mean(l2)),
        "velocity_proxy_rmse_m_per_s": velocity,
        "deformation_gradient_proxy_rmse": deformation,
        "finite_positions": bool(np.isfinite(pred_q).all()),
    }


def velocity_proxy(pred_q: np.ndarray, rollout: dict[str, Any]) -> float | None:
    if "particle_qd" not in rollout or pred_q.shape[0] < 2:
        return None
    dt = np.diff(rollout["time"]).reshape(-1, 1, 1)
    vel = np.diff(pred_q, axis=0) / np.maximum(dt, 1.0e-12)
    target = rollout["particle_qd"][1:]
    return float(np.sqrt(np.mean(np.sum((vel - target) ** 2, axis=-1))))


def deformation_proxy(
    pred_q: np.ndarray, target_q: np.ndarray, rest: np.ndarray, probe_count: int = 256, neighbors: int = 4
) -> float:
    rng = np.random.default_rng(1234)
    count = rest.shape[0]
    probes = rng.choice(count, size=min(probe_count, count), replace=False)
    rest64 = rest.astype(np.float64)
    total = 0.0
    terms = 0
    for p in probes:
        dist = np.sum((rest64 - rest64[p]) ** 2, axis=1)
        nn = np.argsort(dist)[1 : neighbors + 1]
        a = (rest64[nn] - rest64[p]).T
        pinv = np.linalg.pinv(a)
        for frame in range(pred_q.shape[0]):
            pred_a = (pred_q[frame, nn] - pred_q[frame, p]).T
            target_a = (target_q[frame, nn] - target_q[frame, p]).T
            fp = pred_a @ pinv
            ft = target_a @ pinv
            total += float(np.sum((fp - ft) ** 2))
            terms += 1
    return float(np.sqrt(total / max(terms, 1)))


def render_comparison(
    output_dir: Path,
    rollout_dir: Path,
    prediction_dir: Path,
    *,
    name: str = "pod_nn_compare",
    fps: int = 12,
    max_points: int = 9000,
) -> Path:
    rollout = load_rollout(rollout_dir)
    pred_q = np.load(prediction_dir / "decoded_particle_q.npy").astype(np.float32)
    target_q = rollout["particle_q"]
    rng = np.random.default_rng(1234)
    count = target_q.shape[1]
    idx = np.arange(count) if count <= max_points else np.sort(rng.choice(count, size=max_points, replace=False))
    frame_dir = output_dir / name / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    all_points = np.concatenate([target_q[:, idx], pred_q[:, idx]], axis=1)
    xy_min = all_points[:, :, :2].min(axis=(0, 1))
    xy_max = all_points[:, :, :2].max(axis=(0, 1))
    pad = np.maximum((xy_max - xy_min) * 0.08, 1.0e-3)
    xy_min -= pad
    xy_max += pad
    for frame in range(target_q.shape[0]):
        img = _scatter_frame(target_q[frame, idx], pred_q[frame, idx], xy_min, xy_max)
        _write_ppm(frame_dir / f"frame_{frame:04d}.ppm", img)

    mp4_path = output_dir / name / f"{name}.mp4"
    gif_path = output_dir / name / f"{name}.gif"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%04d.ppm"),
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%04d.ppm"), str(gif_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return output_dir / name


def _scatter_frame(target: np.ndarray, pred: np.ndarray, xy_min: np.ndarray, xy_max: np.ndarray) -> np.ndarray:
    width = 960
    height = 540
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_points(img, target, xy_min, xy_max, np.array([40, 90, 210], dtype=np.uint8))
    _draw_points(
        img,
        pred + np.array([0.0, 1.25, 0.0], dtype=np.float32),
        xy_min + np.array([0.0, -0.2]),
        xy_max + np.array([0.0, 1.25]),
        np.array([220, 80, 40], dtype=np.uint8),
    )
    return img


def _draw_points(
    img: np.ndarray, points: np.ndarray, xy_min: np.ndarray, xy_max: np.ndarray, color: np.ndarray
) -> None:
    xy = points[:, :2]
    norm = (xy - xy_min.reshape(1, 2)) / np.maximum((xy_max - xy_min).reshape(1, 2), 1.0e-6)
    px = np.clip((norm[:, 0] * (img.shape[1] - 1)).astype(int), 0, img.shape[1] - 1)
    py = np.clip(((1.0 - norm[:, 1]) * (img.shape[0] - 1)).astype(int), 0, img.shape[0] - 1)
    for x, y in zip(px, py, strict=True):
        img[max(y - 1, 0) : min(y + 2, img.shape[0]), max(x - 1, 0) : min(x + 2, img.shape[1])] = color


def _write_ppm(path: Path, img: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(f"P6\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        f.write(img.tobytes())
