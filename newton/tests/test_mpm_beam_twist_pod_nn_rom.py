# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.mpm.rom.beam_twist_pod_nn import (
    SCHEMA_VERSION,
    control_features,
    controlled_motion,
    fit_pod,
    latent_feature_count,
    project_to_pod,
    rollout_latent_linear,
    rollout_latent_nn,
    train_linear_latent,
    train_mlp,
    write_json,
)


def _write_synthetic_rollout(root: Path, name: str, *, frames: int, twist_speed: float, vertical_amp: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    rest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    time = np.linspace(0.0, 0.1, frames + 1, dtype=np.float64)
    particle_q = np.empty((frames + 1, rest.shape[0], 3), dtype=np.float32)
    for i, t in enumerate(time):
        angle = twist_speed * t
        shear = 0.03 * np.sin(angle)
        lift = vertical_amp * (np.sin(2.0 * np.pi * t) - 0.0)
        particle_q[i] = rest + np.array([0.02 * angle, shear, lift], dtype=np.float32) * rest[:, 0:1]
    dt = float(time[1] - time[0])
    particle_qd = np.zeros_like(particle_q)
    particle_qd[1:] = (particle_q[1:] - particle_q[:-1]) / dt
    particle_qd[0] = particle_qd[1]
    np.save(run_dir / "particle_q.npy", particle_q)
    np.save(run_dir / "particle_qd.npy", particle_qd)
    np.save(run_dir / "rest_position.npy", rest)
    np.save(run_dir / "time.npy", time)
    np.save(run_dir / "twist_indices.npy", np.array([1, 3, 5, 7], dtype=np.int32))
    write_json(
        run_dir / "manifest.json",
        {
            "schema": f"{SCHEMA_VERSION}.teacher_rollout",
            "run_name": name,
            "particle_count": int(rest.shape[0]),
            "frames": frames,
            "config": {
                "device": "cpu",
                "grid_type": "dense",
                "young_modulus": 5.0e6,
                "damping": 0.001,
                "twist_frames": frames,
                "voxel_size": 1.0,
                "fps": 240.0,
                "frame_dt": dt,
                "sim_dt": dt,
                "solver": "synthetic",
                "control_mode": "twist_translate",
                "twist_speed_scale": twist_speed,
                "base_twist_speed": 1.0,
                "twist_speed": twist_speed,
                "lateral_amp": 0.0,
                "vertical_amp": vertical_amp,
                "axial_amp": 0.0,
                "lateral_freq_hz": 0.7,
                "vertical_freq_hz": 1.0,
                "axial_freq_hz": 0.5,
                "lateral_phase": 0.0,
                "vertical_phase": 0.0,
                "axial_phase": 0.0,
            },
            "timing": {
                "step_mean_ms_per_frame": 2.0,
                "step_fps": 500.0,
            },
            "files": {
                "particle_q": "particle_q.npy",
                "particle_qd": "particle_qd.npy",
                "rest_position": "rest_position.npy",
                "time": "time.npy",
                "twist_indices": "twist_indices.npy",
            },
        },
    )
    return run_dir


class TestMpmBeamTwistPodNnRom(unittest.TestCase):
    def test_controlled_motion_starts_from_zero_translation(self):
        config = {
            "twist_speed": 3.0,
            "lateral_amp": 0.2,
            "vertical_amp": 0.3,
            "axial_amp": 0.1,
            "lateral_freq_hz": 0.7,
            "vertical_freq_hz": 0.9,
            "axial_freq_hz": 0.5,
            "lateral_phase": 0.4,
            "vertical_phase": 0.8,
            "axial_phase": 1.2,
        }
        angle, speed, translation, translation_vel = controlled_motion(config, 0.0)
        self.assertEqual(angle, 0.0)
        self.assertEqual(speed, 3.0)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1.0e-7)
        self.assertEqual(translation_vel.shape, (3,))

    def test_synthetic_pod_latent_pipeline_writes_finite_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train0 = _write_synthetic_rollout(root / "teacher", "train0", frames=8, twist_speed=0.9, vertical_amp=0.08)
            train1 = _write_synthetic_rollout(root / "teacher", "train1", frames=8, twist_speed=1.1, vertical_amp=0.12)
            val = _write_synthetic_rollout(root / "teacher", "val", frames=8, twist_speed=1.0, vertical_amp=0.10)

            pod_dir = fit_pod(root / "pod", [train0, train1], rank=3)
            linear_dir = train_linear_latent(root / "models", [train0, train1], pod_dir)
            nn_dir = train_mlp(root / "models", [train0, train1], pod_dir, hidden_dim=8, epochs=3, batch_size=4)
            linear_rollout = rollout_latent_linear(root / "reduced", val, pod_dir, linear_dir)
            nn_rollout = rollout_latent_nn(root / "reduced", val, pod_dir, nn_dir)

            self.assertEqual(latent_feature_count([train0]), 22)
            self.assertEqual(
                control_features(
                    {
                        "time": np.array([0.0]),
                        "manifest": {"config": {"twist_speed": 1.0, "young_modulus": 5.0e6, "twist_speed_scale": 1.0}},
                    }
                ).shape[1],
                7,
            )
            for rollout_dir in (linear_rollout, nn_rollout):
                pred_q = np.load(rollout_dir / "decoded_particle_q.npy")
                self.assertTrue(np.isfinite(pred_q).all())
                self.assertEqual(pred_q.shape, (9, 8, 3))

            pod = np.load(pod_dir / "basis.npy")
            self.assertEqual(pod.shape, (3, 24))
            coeffs = project_to_pod(
                np.load(val / "particle_q.npy"),
                np.load(pod_dir / "rest_position.npy"),
                np.load(pod_dir / "mean_delta.npy"),
                pod,
            )
            self.assertEqual(coeffs.shape, (9, 3))


if __name__ == "__main__":
    unittest.main()
