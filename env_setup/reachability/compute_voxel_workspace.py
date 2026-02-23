#!/usr/bin/env python3
#
# Voxelize workspace point clouds using a convex hull test.
#
# Usage:
#     python compute_voxel_workspace.py
#

import os
import sys
import numpy as np

try:
    from scipy.spatial import ConvexHull, Delaunay
except ImportError as exc:
    print(f"[ERROR] scipy is required: {exc}", file=sys.stderr)
    sys.exit(1)


def _write_ply(path, points):
    with open(path, "w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for pt in points:
            handle.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")


def _points_in_hull(points, grid_points):
    if points.shape[0] < 4:
        return np.zeros((grid_points.shape[0],), dtype=bool)
    centered = points - points.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 3:
        return np.zeros((grid_points.shape[0],), dtype=bool)
    hull = ConvexHull(points)
    delaunay = Delaunay(points[hull.vertices])
    return delaunay.find_simplex(grid_points) >= 0


def main():
    class Config:
        pass

    cfg = Config()
    cfg.input_npz = "/home/user/rl_ws/env_setup/reachability/output/openarm_tesollo_t1_workspace_points.npz"
    cfg.output_dir = "/home/user/rl_ws/env_setup/reachability/output"
    cfg.output_prefix = "openarm_tesollo_t1"
    cfg.voxels_per_dim = 30
    cfg.padding = 0.1
    cfg.save_ply = True
    cfg.voxel_stride = 1
    cfg.symmetrize = True
    cfg.symmetrize_mode = "intersection"
    cfg.mirror_axis = "y"
    cfg.mirror_center = 0.0
    cfg.left_ee_name = "ll_dg_ee"
    cfg.right_ee_name = "rl_dg_ee"
    cfg.compute_z_axis_bins = True
    cfg.z_axis_bins_yaw = 12
    cfg.z_axis_bins_pitch = 6
    cfg.z_axis_min_count = 1

    if not os.path.isfile(cfg.input_npz):
        print(f"[ERROR] Input file not found: {cfg.input_npz}", file=sys.stderr)
        return

    os.makedirs(cfg.output_dir, exist_ok=True)

    data = np.load(cfg.input_npz, allow_pickle=True)
    ee_body_names = data["ee_body_names"].tolist()
    points = data["points"]
    z_axes_b = data["z_axes_b"] if "z_axes_b" in data else None

    all_points = points.reshape(-1, 3)
    grid_min = all_points.min(axis=0) - cfg.padding
    grid_max = all_points.max(axis=0) + cfg.padding
    if cfg.symmetrize:
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis_map.get(str(cfg.mirror_axis).lower())
        if axis is None:
            raise ValueError(f"Unsupported mirror_axis: {cfg.mirror_axis}")
        max_dist = max(abs(grid_min[axis] - cfg.mirror_center), abs(grid_max[axis] - cfg.mirror_center))
        grid_min[axis] = cfg.mirror_center - max_dist
        grid_max[axis] = cfg.mirror_center + max_dist

    xs = np.linspace(grid_min[0], grid_max[0], cfg.voxels_per_dim)
    ys = np.linspace(grid_min[1], grid_max[1], cfg.voxels_per_dim)
    zs = np.linspace(grid_min[2], grid_max[2], cfg.voxels_per_dim)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    grid_points = grid.reshape(-1, 3)

    reach_mesh = np.zeros((len(ee_body_names), cfg.voxels_per_dim, cfg.voxels_per_dim, cfg.voxels_per_dim), dtype=np.uint8)
    for idx, name in enumerate(ee_body_names):
        inside = _points_in_hull(points[:, idx, :], grid_points)
        reach_mesh[idx] = inside.reshape(cfg.voxels_per_dim, cfg.voxels_per_dim, cfg.voxels_per_dim).astype(np.uint8)

    if cfg.symmetrize:
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis_map.get(str(cfg.mirror_axis).lower())
        if cfg.left_ee_name in ee_body_names and cfg.right_ee_name in ee_body_names:
            left_idx = ee_body_names.index(cfg.left_ee_name)
            right_idx = ee_body_names.index(cfg.right_ee_name)
            left_mask = reach_mesh[left_idx].astype(bool)
            right_mask = reach_mesh[right_idx].astype(bool)
            mirrored_right = np.flip(right_mask, axis=axis)
            if cfg.symmetrize_mode == "intersection":
                common = left_mask & mirrored_right
            elif cfg.symmetrize_mode == "union":
                common = left_mask | mirrored_right
            else:
                raise ValueError(f"Unsupported symmetrize_mode: {cfg.symmetrize_mode}")
            reach_mesh[left_idx] = common.astype(np.uint8)
            reach_mesh[right_idx] = np.flip(common, axis=axis).astype(np.uint8)
        else:
            print("[WARN] Symmetrize enabled, but left/right EE names not found.", file=sys.stderr)

    if cfg.save_ply:
        for idx, name in enumerate(ee_body_names):
            inside_idx = np.argwhere(reach_mesh[idx] > 0)
            if cfg.voxel_stride > 1:
                inside_idx = inside_idx[:: cfg.voxel_stride]
            if inside_idx.size > 0:
                voxel_points = np.column_stack((xs[inside_idx[:, 0]], ys[inside_idx[:, 1]], zs[inside_idx[:, 2]]))
                output_ply = os.path.join(cfg.output_dir, f"{cfg.output_prefix}_{name}_voxels.ply")
                _write_ply(output_ply, voxel_points)
                print(f"[INFO] Saved voxel PLY: {output_ply}")

    output_npz = os.path.join(cfg.output_dir, f"{cfg.output_prefix}_voxels.npz")
    output_payload = dict(
        ee_body_names=ee_body_names,
        reach_mesh=reach_mesh,
        grid_min=grid_min,
        grid_max=grid_max,
        voxels_per_dim=cfg.voxels_per_dim,
    )
    if cfg.compute_z_axis_bins and z_axes_b is not None:
        if z_axes_b.shape[:2] != points.shape[:2]:
            print("[WARN] z_axes_b shape does not match points; skipping z-axis bins.", file=sys.stderr)
        else:
            spacing = (grid_max - grid_min) / max(cfg.voxels_per_dim - 1, 1)
            yaw_edges = np.linspace(-np.pi, np.pi, cfg.z_axis_bins_yaw + 1, dtype=np.float32)
            pitch_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, cfg.z_axis_bins_pitch + 1, dtype=np.float32)
            z_axis_bins = np.zeros(
                (
                    len(ee_body_names),
                    cfg.voxels_per_dim,
                    cfg.voxels_per_dim,
                    cfg.voxels_per_dim,
                    cfg.z_axis_bins_yaw,
                    cfg.z_axis_bins_pitch,
                ),
                dtype=np.uint8,
            )
            for idx, name in enumerate(ee_body_names):
                pts = points[:, idx, :]
                dirs = z_axes_b[:, idx, :]
                norms = np.linalg.norm(dirs, axis=1, keepdims=True)
                dirs = dirs / np.clip(norms, 1.0e-8, None)

                coords = np.rint((pts - grid_min) / spacing).astype(np.int64)
                coords = np.clip(coords, 0, cfg.voxels_per_dim - 1)

                yaw = np.arctan2(dirs[:, 1], dirs[:, 0])
                pitch = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
                yaw_idx = np.digitize(yaw, yaw_edges) - 1
                pitch_idx = np.digitize(pitch, pitch_edges) - 1
                yaw_idx = np.clip(yaw_idx, 0, cfg.z_axis_bins_yaw - 1)
                pitch_idx = np.clip(pitch_idx, 0, cfg.z_axis_bins_pitch - 1)

                if cfg.z_axis_min_count > 1:
                    counts = np.zeros(
                        (
                            cfg.voxels_per_dim,
                            cfg.voxels_per_dim,
                            cfg.voxels_per_dim,
                            cfg.z_axis_bins_yaw,
                            cfg.z_axis_bins_pitch,
                        ),
                        dtype=np.uint16,
                    )
                    np.add.at(
                        counts,
                        (coords[:, 0], coords[:, 1], coords[:, 2], yaw_idx, pitch_idx),
                        1,
                    )
                    z_axis_bins[idx] = (counts >= cfg.z_axis_min_count).astype(np.uint8)
                else:
                    z_axis_bins[idx, coords[:, 0], coords[:, 1], coords[:, 2], yaw_idx, pitch_idx] = 1

            output_payload.update(
                z_axis_bins=z_axis_bins,
                z_axis_yaw_edges=yaw_edges,
                z_axis_pitch_edges=pitch_edges,
            )
            print("[INFO] Saved z-axis orientation bins in voxel data.")
    elif cfg.compute_z_axis_bins:
        print("[WARN] z_axes_b not found; skipping z-axis bins.", file=sys.stderr)

    np.savez_compressed(output_npz, **output_payload)
    print(f"[INFO] Saved voxel data: {output_npz}")


if __name__ == "__main__":
    main()
