#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render MPM beam particle rollouts as an animated surface mesh.

Run with Blender:

    blender --background --python tools/render_mpm_beam_mesh_blender.py -- \
      --teacher-rollout artifacts/.../teacher_rollouts/val_twist_1p0 \
      --prediction-rollout artifacts/.../reduced_rollouts/val_twist_1p0_pod_nn \
      --output-dir artifacts/.../mesh_renders/twist_1p0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--teacher-rollout", type=Path, required=True)
    parser.add_argument("--prediction-rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="twist_1p0")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args(argv)


def load_arrays(teacher_dir: Path, prediction_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    teacher_q = np.load(teacher_dir / "particle_q.npy").astype(np.float32)
    prediction_q = np.load(prediction_dir / "decoded_particle_q.npy").astype(np.float32)
    report = json.loads((prediction_dir / "rollout_report.json").read_text())
    frames = min(teacher_q.shape[0], prediction_q.shape[0])
    return teacher_q[:frames], prediction_q[:frames], report


def infer_surface_faces(rest: np.ndarray) -> list[tuple[int, int, int, int]]:
    rounded = np.round(rest, 6)
    xs = sorted(set(rounded[:, 0]))
    ys = sorted(set(rounded[:, 1]))
    zs = sorted(set(rounded[:, 2]))
    index = {(p[0], p[1], p[2]): i for i, p in enumerate(rounded)}

    def vid(i: int, j: int, k: int) -> int:
        return index[(xs[i], ys[j], zs[k])]

    nx, ny, nz = len(xs), len(ys), len(zs)
    faces: list[tuple[int, int, int, int]] = []

    for i in (0, nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                faces.append((vid(i, j, k), vid(i, j + 1, k), vid(i, j + 1, k + 1), vid(i, j, k + 1)))
    for j in (0, ny - 1):
        for i in range(nx - 1):
            for k in range(nz - 1):
                faces.append((vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j, k + 1), vid(i, j, k + 1)))
    for k in (0, nz - 1):
        for i in range(nx - 1):
            for j in range(ny - 1):
                faces.append((vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k)))

    return faces


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def make_mesh_object(
    name: str,
    vertices: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    material: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def update_mesh(obj: bpy.types.Object, vertices: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", vertices.reshape(-1))
    obj.data.update()


def look_at(obj: bpy.types.Object, target: np.ndarray) -> None:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(obj.location, dtype=np.float64)
    obj.rotation_euler = direction_to_euler(direction)


def direction_to_euler(direction: np.ndarray) -> tuple[float, float, float]:
    quat = mathutils.Vector(direction).to_track_quat("-Z", "Y")
    return quat.to_euler()


def setup_camera_and_lights(all_points: np.ndarray, width: int, height: int, fps: int) -> None:
    center = all_points.mean(axis=0)
    span = np.maximum(all_points.max(axis=0) - all_points.min(axis=0), 1.0e-3)

    camera = bpy.data.cameras.new("Camera")
    camera.type = "ORTHO"
    camera.ortho_scale = float(max(span[0] * 1.18, span[2] * 1.55, 3.8))
    cam_obj = bpy.data.objects.new("Camera", camera)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (float(center[0] + 2.5), float(center[1] - 7.0), float(center[2] + 2.6))
    look_at(cam_obj, center + np.array([0.0, 0.0, 0.05]))
    bpy.context.scene.camera = cam_obj

    bpy.ops.object.light_add(
        type="AREA", location=(float(center[0] - 2.0), float(center[1] - 4.0), float(center[2] + 5.0))
    )
    light = bpy.context.object
    light.name = "KeyLight"
    light.data.energy = 450.0
    light.data.size = 5.0

    bpy.ops.object.light_add(
        type="POINT", location=(float(center[0] + 3.0), float(center[1] + 2.0), float(center[2] + 3.0))
    )
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 95.0

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 64
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.fps = fps
    scene.world.color = (0.97, 0.97, 0.95)


def write_usda(
    path: Path, teacher_q: np.ndarray, pred_q: np.ndarray, faces: list[tuple[int, int, int, int]], shift: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = ", ".join(["4"] * len(faces))
    indices = ", ".join(str(v) for face in faces for v in face)

    def points_samples(q: np.ndarray, z_shift: float) -> str:
        lines = []
        for frame, pts in enumerate(q):
            shifted = pts.copy()
            shifted[:, 2] += z_shift
            point_text = ", ".join(f"({p[0]:.7g}, {p[1]:.7g}, {p[2]:.7g})" for p in shifted)
            lines.append(f"            {frame}: [{point_text}]")
        return ",\n".join(lines)

    content = f"""#usda 1.0
(
    defaultPrim = "World"
    startTimeCode = 0
    endTimeCode = {teacher_q.shape[0] - 1}
    framesPerSecond = 12
    timeCodesPerSecond = 12
    upAxis = "Z"
)

def Xform "World"
{{
    def Mesh "NewtonTeacher"
    {{
        uniform bool doubleSided = 1
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points.timeSamples = {{
{points_samples(teacher_q, shift)}
        }}
        color3f[] primvars:displayColor = [(0.10, 0.32, 0.90)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}

    def Mesh "POD_NN"
    {{
        uniform bool doubleSided = 1
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points.timeSamples = {{
{points_samples(pred_q, -shift)}
        }}
        color3f[] primvars:displayColor = [(0.95, 0.33, 0.12)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
}}
"""
    path.write_text(content)


def main() -> None:
    args = parse_args()
    teacher_q, pred_q, report = load_arrays(args.teacher_rollout, args.prediction_rollout)
    faces = infer_surface_faces(teacher_q[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    z_shift = 1.05
    write_usda(args.output_dir / f"{args.label}_teacher_vs_pod_nn_mesh.usda", teacher_q, pred_q, faces, z_shift)

    reset_scene()
    teacher_mat = make_material("NewtonTeacherBlue", (0.05, 0.22, 0.95, 1.0))
    pred_mat = make_material("PodNnOrange", (0.95, 0.28, 0.08, 1.0))
    teacher = make_mesh_object("NewtonTeacher", teacher_q[0] + np.array([0.0, 0.0, z_shift]), faces, teacher_mat)
    pred = make_mesh_object("POD_NN", pred_q[0] + np.array([0.0, 0.0, -z_shift]), faces, pred_mat)

    all_points = np.concatenate(
        [
            teacher_q.reshape(-1, 3) + np.array([0.0, 0.0, z_shift]),
            pred_q.reshape(-1, 3) + np.array([0.0, 0.0, -z_shift]),
        ],
        axis=0,
    )
    setup_camera_and_lights(all_points, args.width, args.height, args.fps)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = teacher_q.shape[0] - 1
    render_index = 0
    for frame in range(0, teacher_q.shape[0], max(args.frame_stride, 1)):
        update_mesh(teacher, teacher_q[frame] + np.array([0.0, 0.0, z_shift]))
        update_mesh(pred, pred_q[frame] + np.array([0.0, 0.0, -z_shift]))
        scene.frame_set(frame)
        scene.render.filepath = str(frame_dir / f"frame_{render_index:04d}.png")
        bpy.ops.render.render(write_still=True)
        render_index += 1

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    mp4 = args.output_dir / f"{args.label}_teacher_vs_pod_nn_mesh.mp4"
    gif = args.output_dir / f"{args.label}_teacher_vs_pod_nn_mesh.gif"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(args.fps), "-i", str(frame_dir / "frame_%04d.png"), str(gif)],
        check=True,
    )

    summary = {
        "label": args.label,
        "teacher_rollout": str(args.teacher_rollout),
        "prediction_rollout": str(args.prediction_rollout),
        "usd": str(args.output_dir / f"{args.label}_teacher_vs_pod_nn_mesh.usda"),
        "mp4": str(mp4),
        "gif": str(gif),
        "quality": report["quality"],
        "timing": report["timing"],
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
