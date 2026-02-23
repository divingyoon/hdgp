# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finger_contact_flags_from_sensor(
    force_magnitudes: torch.Tensor,
    contact_threshold: float,
    sensor_body_names: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Aggregate link-level contact forces into finger-level boolean flags.

    For the T3 left hand, preferred mapping is by sensor link names:
    - finger_1: seg3, tip
    - finger_2/3/4: seg2, seg3, tip
    - finger_5: seg3, tip
    plus 3 palm sensors (ignored for finger coverage).
    """
    num_links = force_magnitudes.shape[1]
    link_flags = force_magnitudes > contact_threshold

    # 1) Preferred: explicit mapping by sensor link names when available.
    if sensor_body_names is not None and len(sensor_body_names) == num_links:
        finger_flags: list[torch.Tensor] = []
        for finger_id in (1, 2, 3, 4, 5):
            idxs = [i for i, name in enumerate(sensor_body_names) if f"finger_{finger_id}_" in str(name)]
            if idxs:
                finger_flags.append(link_flags[:, idxs].any(dim=1))
            else:
                finger_flags.append(torch.zeros(link_flags.shape[0], device=link_flags.device, dtype=torch.bool))
        return torch.stack(finger_flags, dim=1)

    # 2) Fallback for known T3 ordering without names:
    # [palm1, palm2, palm3, f1(2), f2(3), f3(3), f4(3), f5(2)] = 16
    if num_links == 16:
        groups = [
            link_flags[:, 3:5],    # finger 1
            link_flags[:, 5:8],    # finger 2
            link_flags[:, 8:11],   # finger 3
            link_flags[:, 11:14],  # finger 4
            link_flags[:, 14:16],  # finger 5
        ]
        return torch.stack([g.any(dim=1) for g in groups], dim=1)

    # 3) Older compact setup: 10 links -> pairwise mapping.
    if num_links >= 10:
        trimmed = link_flags[:, :10]
        finger_flags = trimmed.reshape(trimmed.shape[0], 5, 2).any(dim=2)
    else:
        # Last-resort fallback: contiguous chunks into up to 5 groups.
        group_count = max(1, min(5, num_links))
        chunk = max(1, num_links // group_count)
        groups: list[torch.Tensor] = []
        for i in range(group_count):
            s = i * chunk
            e = num_links if i == group_count - 1 else min(num_links, (i + 1) * chunk)
            if s >= num_links:
                groups.append(torch.zeros(link_flags.shape[0], device=link_flags.device, dtype=torch.bool))
            else:
                groups.append(link_flags[:, s:e].any(dim=1))
        finger_flags = torch.stack(groups, dim=1)

    return finger_flags


def _select_sensor_body_names(
    sensor_body_names: list[str] | tuple[str, ...] | None,
    body_ids,
) -> list[str] | tuple[str, ...] | None:
    """Select sensor body names using body_ids that may be slice/list/tensor."""
    if sensor_body_names is None or body_ids is None:
        return sensor_body_names
    if isinstance(body_ids, slice):
        return sensor_body_names[body_ids]
    if torch.is_tensor(body_ids):
        body_ids = body_ids.tolist()
    return [sensor_body_names[int(i)] for i in body_ids]


def _sensor_force_magnitudes_filtered(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, list[str] | tuple[str, ...] | None]:
    """Return per-body contact force magnitudes, preferring filtered object-contact matrix.

    When ``filter_prim_paths_expr`` is provided in ContactSensorCfg, ``force_matrix_w`` contains
    contacts only against those filtered prims (Cup/Cup2 in this task). Using ``net_forces_w``
    would include table/self contacts and can contaminate grasp rewards.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    sensor_body_names = getattr(contact_sensor, "body_names", None)
    if sensor_body_names is None:
        sensor_body_names = getattr(contact_sensor.data, "body_names", None)

    force_matrix_w = getattr(contact_sensor.data, "force_matrix_w", None)
    if force_matrix_w is not None:
        # (N, B, F, 3) -> (N, B): max filtered-contact magnitude per sensor body
        force_magnitudes = torch.norm(force_matrix_w, dim=-1).max(dim=-1)[0]
    else:
        # Strict mode: disable contact reward if filtered matrix is unavailable.
        strict_filtered_only = bool(getattr(getattr(env, "cfg", None), "require_filtered_contact_matrix", False))
        if strict_filtered_only:
            force_magnitudes = torch.zeros_like(torch.norm(contact_sensor.data.net_forces_w, dim=-1))
        else:
            # Fallback: unfiltered net forces (may include table/self contacts).
            force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]
        sensor_body_names = _select_sensor_body_names(sensor_body_names, sensor_cfg.body_ids)

    return force_magnitudes, sensor_body_names


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w)
    return obj_pos_b


def _get_episode_initial_object_xy(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    cache_attr: str,
) -> torch.Tensor:
    """Track per-episode initial object XY in world frame."""
    current_xy = obj.data.root_pos_w[:, :2]
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, current_xy.clone())
    initial_xy = getattr(env, cache_attr)
    reset_mask = (env.episode_length_buf == 0).squeeze(-1)
    initial_xy[reset_mask] = current_xy[reset_mask]
    setattr(env, cache_attr, initial_xy)
    return initial_xy


def _compute_grasp_target_pos_w(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    ee_pos_w: torch.Tensor,
    use_dynamic_z: bool,
    dynamic_z_state_attr: str | None = None,
) -> torch.Tensor:
    """Compute grasp target from grasp2g offset in world frame."""
    cfg = getattr(env, "cfg", None)
    obj_pos_w = obj.data.root_pos_w.clone()
    base_offset = getattr(cfg, "grasp2g_target_offset", (0.0, 0.0, 0.08))

    if not (isinstance(base_offset, (list, tuple)) and len(base_offset) == 3):
        return obj_pos_w

    offset_xy_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
    offset_xy_local[:, 0] = base_offset[0]
    offset_xy_local[:, 1] = base_offset[1]
    offset_xy_w = quat_apply(obj.data.root_quat_w, offset_xy_local)
    target_xy_w = obj_pos_w[:, :2] + offset_xy_w[:, :2]

    z_value: torch.Tensor | float = base_offset[2]
    if use_dynamic_z:
        # Use XY distance to the offset target (not cup center) for z transition.
        xy_dist = torch.norm(ee_pos_w[:, :2] - target_xy_w, dim=1)
        z_high = float(getattr(cfg, "reach_dynamic_z_high", 0.2))
        x_hi = float(getattr(cfg, "reach_dynamic_xy_hi", 0.06))
        x_lo = float(getattr(cfg, "reach_dynamic_xy_lo", 0.015))
        x_gate = float(getattr(cfg, "reach_dynamic_xy_gate", x_hi))
        x_gate = min(x_gate, x_hi)
        x_gate = max(x_gate, x_lo + 1e-6)
        x_hi = max(x_hi, x_lo + 1e-6)

        # Keep high-Z approach until XY is close enough, then start descending.
        u = torch.clamp((x_gate - xy_dist) / (x_gate - x_lo), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)  # smoothstep
        z_value_raw = z_high * (1.0 - u) + float(base_offset[2]) * u

        # Limit per-step Z descent speed to avoid sudden dives toward the cup.
        descent_rate = float(getattr(cfg, "reach_dynamic_z_descent_rate", 0.0))
        if descent_rate > 0.0 and dynamic_z_state_attr is not None:
            if not hasattr(env, dynamic_z_state_attr):
                setattr(env, dynamic_z_state_attr, torch.full_like(z_value_raw, z_high))
            z_prev = getattr(env, dynamic_z_state_attr)
            reset_mask = (env.episode_length_buf == 0).squeeze(-1)
            z_prev[reset_mask] = z_high

            z_floor = z_prev - descent_rate
            z_value = torch.maximum(z_value_raw, z_floor)
            setattr(env, dynamic_z_state_attr, z_value)
        else:
            z_value = z_value_raw

    offset_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
    offset_local[:, 0] = base_offset[0]
    offset_local[:, 1] = base_offset[1]
    if isinstance(z_value, torch.Tensor):
        offset_local[:, 2] = z_value
    else:
        offset_local[:, 2] = float(z_value)
    offset_w = quat_apply(obj.data.root_quat_w, offset_local)
    return obj_pos_w + offset_w


def _stage_lambda(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """DexPour λ trigger: approach complete."""
    cfg = getattr(env, "cfg", None)
    threshold = float(getattr(cfg, "dexpour_approach_threshold", 0.05))
    return _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=threshold)


def _stage_mu(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_contact_sensor"),
) -> torch.Tensor:
    """DexPour μ trigger: secure grasp complete after approach."""
    cfg = getattr(env, "cfg", None)
    min_contacts = int(getattr(cfg, "dexpour_grasp_min_contacts", 4))
    min_contacts = max(1, min(5, min_contacts))
    contact_threshold = float(getattr(cfg, "dexpour_contact_threshold", 0.02))
    require_thumb = bool(getattr(cfg, "dexpour_require_thumb_contact", True))

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    force_magnitudes, sensor_body_names = _sensor_force_magnitudes_filtered(env, sensor_cfg)
    finger_flags = _finger_contact_flags_from_sensor(force_magnitudes, contact_threshold, sensor_body_names)
    num_contacts = finger_flags.sum(dim=-1)
    contact_ok = (num_contacts >= min_contacts).float()
    if require_thumb:
        contact_ok = contact_ok * finger_flags[:, 0].float()
    return lambda_trigger * contact_ok


def _stage_nu(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_contact_sensor"),
) -> torch.Tensor:
    """DexPour ν trigger: cup lift complete after secure grasp."""
    mu_trigger = _stage_mu(env, object_cfg, eef_link_name, sensor_cfg=sensor_cfg)
    stable_lift = _is_transport_stably_complete(env, object_cfg)
    return mu_trigger * stable_lift


def _debug_stage_triggers(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str = "ll_dg_ee",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_contact_sensor"),
) -> None:
    """Print DexPour stage triggers (env 0 only) for terminal debugging."""
    cfg = getattr(env, "cfg", None)
    debug_enabled = bool(getattr(cfg, "debug_stage_triggers", True))
    interval = int(getattr(cfg, "debug_stage_triggers_interval", 50))
    interval = max(1, interval)
    step_count = int(getattr(env, "common_step_counter", -1))
    if (not debug_enabled) or (step_count < 0) or (step_count % interval != 0):
        return

    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    lambda_t = _stage_lambda(env, object_cfg, eef_link_name)
    mu_t = _stage_mu(env, object_cfg, eef_link_name, sensor_cfg=sensor_cfg)
    nu_t = _stage_nu(env, object_cfg, eef_link_name, sensor_cfg=sensor_cfg)
    finger_flags = _left_finger_contact_flags(env, sensor_cfg=sensor_cfg, object_cfg=object_cfg, contact_threshold=0.02)
    contacts = finger_flags.sum(dim=1).float()
    lambda_threshold = float(getattr(cfg, "dexpour_approach_threshold", 0.05))

    print(
        f"[Step {step_count}] "
        f"dist={dist[0].item():.4f}m(th={lambda_threshold:.3f}) | "
        f"contacts={contacts[0].item():.0f}/5 | "
        f"lambda={lambda_t[0].item():.0f} mu={mu_t[0].item():.0f} nu={nu_t[0].item():.0f} | "
        f"cup_z={obj.data.root_pos_w[0, 2].item():.3f}"
    )


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.

    Uses dynamic z offset: starts high (0.15) to approach from above,
    then lowers to grasp position (0.08) as xy alignment improves.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # Get EE position first for dynamic offset calculation
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(
        env,
        obj,
        ee_pos_w,
        use_dynamic_z=True,
        dynamic_z_state_attr="_reach_dynamic_z_prev_left",
    )
    _maybe_visualize_approach_target_all(env, target_pos_w, obj.data.root_quat_w, marker_attr="_debug_approach_target_left")

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)

    # Suppress reaching reward when cup is pushed in XY during approach.
    cfg = getattr(env, "cfg", None)
    disp_free = float(getattr(cfg, "reach_displacement_free_threshold", 0.005))
    disp_scale = float(getattr(cfg, "reach_displacement_suppress_scale", 0.01))
    current_xy = obj.data.root_pos_w[:, :2]
    initial_xy = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")
    displacement_xy = torch.norm(current_xy - initial_xy, dim=1)
    displacement_excess = torch.clamp(displacement_xy - disp_free, min=0.0)
    if disp_scale > 0.0:
        reach_reward = reach_reward * torch.exp(-displacement_excess / disp_scale)

    # Stage-1 reward: active while λ == 0.
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    _debug_stage_triggers(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reach_reward


def object_ee_distance_fine(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Fine-grained reaching reward toward the static grasp target (use_dynamic_z=False).

    Provides gradient to guide the EE from the dynamic target position
    all the way down to the actual grasp position (Z=offset[2]).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(
        env, obj, ee_pos_w, use_dynamic_z=False,
    )

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)

    # Stage-1 reward: active while λ == 0.
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reach_reward


def _maybe_visualize_approach_target_all(
    env: ManagerBasedRLEnv,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    marker_attr: str,
) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_approach_target_vis", True):
        return

    interval = int(getattr(cfg, "debug_approach_target_vis_interval", 10))
    step_count = int(getattr(env, "common_step_counter", 0))
    if interval > 1 and (step_count % interval) != 0:
        return

    if not hasattr(env, marker_attr):
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/ApproachTargetLeft")
        marker_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
        marker = VisualizationMarkers(marker_cfg)
        marker.set_visibility(True)
        setattr(env, marker_attr, marker)

    marker = getattr(env, marker_attr)
    marker.visualize(target_pos_w, target_quat_w)


def _object_eef_any_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Maximum absolute alignment between any EE axis and any object axis."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    eef_axes = [quat_apply(eef_quat, x_axis), quat_apply(eef_quat, y_axis), quat_apply(eef_quat, z_axis)]
    obj_axes = [quat_apply(object_quat, x_axis), quat_apply(object_quat, y_axis), quat_apply(object_quat, z_axis)]

    max_align = torch.zeros(env.num_envs, device=env.device)
    for eef_axis in eef_axes:
        for obj_axis in obj_axes:
            align = torch.abs(torch.sum(eef_axis * obj_axis, dim=1))
            max_align = torch.maximum(max_align, align)
    return max_align


def eef_to_object_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward the end-effector aligning with any object axis (loose tanh-kernel)."""
    max_align = _object_eef_any_axis_alignment(env, eef_link_name, object_cfg)
    error = 1.0 - max_align
    return 1 - torch.tanh(error / std)


def eef_z_perpendicular_object_z(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str = "ll_dg_ee",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward 90-degree alignment between EE +Z axis and object +Z axis."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat.dtype).repeat(env.num_envs, 1)
    ee_z = quat_apply(eef_quat, z_axis)
    obj_z = quat_apply(object_quat, z_axis)

    cos_theta = torch.sum(ee_z * obj_z, dim=1).clamp(-1.0, 1.0)
    error = torch.abs(cos_theta)
    orientation_reward = 1 - torch.tanh(error / std)
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * orientation_reward


def _is_reaching_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    reach_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if EE has reached the grasp position (z_low offset near object).

    Returns a boolean mask (float 0/1) per environment.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    return (dist < reach_threshold).float()


def _reaching_soft_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Continuous [0, 1] gate based on EE distance to static grasp target."""
    cfg = getattr(env, "cfg", None)
    near = float(getattr(cfg, "reach_soft_gate_near", 0.02))
    far = float(getattr(cfg, "reach_soft_gate_far", 0.08))
    far = max(far, near + 1e-6)

    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    gate = torch.clamp((far - dist) / (far - near), 0.0, 1.0)
    return gate


def _is_reaching_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Gate progression after reaching is maintained for multiple consecutive steps."""
    cfg = getattr(env, "cfg", None)
    reach_threshold = float(getattr(cfg, "reach_switch_threshold", 0.01))
    hold_steps = int(getattr(cfg, "reach_switch_hold_steps", 10))
    hold_steps = max(1, hold_steps)

    reached_now = _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=reach_threshold)
    reached_now_i64 = reached_now.to(dtype=torch.int64)

    if not hasattr(env, "_reach_hold_counter_left"):
        env._reach_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._reach_hold_counter_left

    # Update once per sim step even if multiple reward terms query this gate.
    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_reach_hold_counter_left_last_step"):
        env._reach_hold_counter_left_last_step = -2
    if env._reach_hold_counter_left_last_step != step_count:
        # Reset counter at episode boundary.
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0

        counter = torch.where(reached_now_i64 > 0, counter + 1, torch.zeros_like(counter))
        env._reach_hold_counter_left = counter
        env._reach_hold_counter_left_last_step = step_count

    return (counter >= hold_steps).to(dtype=reached_now.dtype)


def _reaching_progress_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Combine soft and stable gates for robust phase transition."""
    stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    soft = _reaching_soft_gate(env, object_cfg, eef_link_name)
    soft_relaxed = torch.clamp(soft * 1.2, 0.0, 1.0)
    return torch.maximum(stable, soft_relaxed)


def _is_grasp_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Gate finger-closing progression after very-close reaching is maintained."""
    cfg = getattr(env, "cfg", None)
    grasp_threshold = float(getattr(cfg, "grasp_switch_threshold", 0.025))
    hold_steps = int(getattr(cfg, "grasp_switch_hold_steps", 2))
    hold_steps = max(1, hold_steps)

    reached_now = _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=grasp_threshold)
    reached_now_i64 = reached_now.to(dtype=torch.int64)

    if not hasattr(env, "_grasp_hold_counter_left"):
        env._grasp_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._grasp_hold_counter_left

    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_grasp_hold_counter_left_last_step"):
        env._grasp_hold_counter_left_last_step = -2
    if env._grasp_hold_counter_left_last_step != step_count:
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0
        counter = torch.where(reached_now_i64 > 0, counter + 1, torch.zeros_like(counter))
        env._grasp_hold_counter_left = counter
        env._grasp_hold_counter_left_last_step = step_count

    return (counter >= hold_steps).to(dtype=reached_now.dtype)


def _grasp_soft_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Continuous [0, 1] gate for grasp activation (closer band than reaching gate)."""
    cfg = getattr(env, "cfg", None)
    near = float(getattr(cfg, "grasp_soft_gate_near", 0.012))
    far = float(getattr(cfg, "grasp_soft_gate_far", 0.03))
    far = max(far, near + 1e-6)

    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    gate = torch.clamp((far - dist) / (far - near), 0.0, 1.0)
    return gate


def _grasp_orientation_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Continuous gate [0,1] from EE/cup pre-grasp orientation quality."""
    cfg = getattr(env, "cfg", None)
    min_reward = float(getattr(cfg, "grasp_orientation_gate_min_reward", 0.25))
    full_reward = float(getattr(cfg, "grasp_orientation_gate_full_reward", 0.75))
    full_reward = max(full_reward, min_reward + 1e-6)

    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat.dtype).repeat(env.num_envs, 1)
    ee_z = quat_apply(eef_quat, z_axis)
    obj_z = quat_apply(object_quat, z_axis)

    cos_theta = torch.sum(ee_z * obj_z, dim=1).clamp(-1.0, 1.0)
    err = torch.abs(cos_theta)
    std = float(getattr(cfg, "grasp_orientation_std", 0.2))
    orient_reward = 1.0 - torch.tanh(err / max(std, 1e-6))

    return torch.clamp((orient_reward - min_reward) / (full_reward - min_reward), 0.0, 1.0)


def _grasp_displacement_safety_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Suppress grasp gate when cup is pushed in XY before secure grasp."""
    cfg = getattr(env, "cfg", None)
    obj: RigidObject = env.scene[object_cfg.name]

    free = float(getattr(cfg, "grasp_displacement_free_threshold", 0.01))
    scale = float(getattr(cfg, "grasp_displacement_suppress_scale", 0.015))
    current_xy = obj.data.root_pos_w[:, :2]
    initial_xy = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")
    displacement_xy = torch.norm(current_xy - initial_xy, dim=1)
    excess = torch.clamp(displacement_xy - free, min=0.0)
    if scale <= 0.0:
        return torch.ones_like(excess)
    return torch.exp(-excess / scale)


def _grasp_progress_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Grasp gate with orientation/safety checks for robust pre-grasp transition."""
    cfg = getattr(env, "cfg", None)
    stable = _is_grasp_stably_complete(env, object_cfg, eef_link_name)
    soft = _grasp_soft_gate(env, object_cfg, eef_link_name)
    soft_prefactor = float(getattr(cfg, "grasp_soft_prefactor", 0.2))
    soft_prefactor = max(0.0, min(1.0, soft_prefactor))
    soft_relaxed = torch.clamp(soft * soft_prefactor, 0.0, 1.0)

    base_gate = torch.maximum(stable, soft_relaxed)
    orientation_gate = _grasp_orientation_gate(env, object_cfg, eef_link_name)
    displacement_gate = _grasp_displacement_safety_gate(env, object_cfg)
    return base_gate * orientation_gate * displacement_gate


def _transport_soft_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Continuous [0,1] gate for transfer stage based on cup height."""
    cfg = getattr(env, "cfg", None)
    h_lo = float(getattr(cfg, "transfer_soft_height_lo", 0.05))
    h_hi = float(getattr(cfg, "transfer_soft_height_hi", 0.10))
    h_hi = max(h_hi, h_lo + 1e-6)

    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    return torch.clamp((z - h_lo) / (h_hi - h_lo), 0.0, 1.0)


def _is_transport_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Binary transfer trigger once cup lift is stably maintained."""
    cfg = getattr(env, "cfg", None)
    h_switch = float(getattr(cfg, "transfer_switch_height", 0.10))
    hold_steps = int(getattr(cfg, "transfer_switch_hold_steps", 4))
    hold_steps = max(1, hold_steps)

    obj: RigidObject = env.scene[object_cfg.name]
    lifted_now = (obj.data.root_pos_w[:, 2] > h_switch).to(dtype=torch.int64)

    if not hasattr(env, "_transfer_hold_counter_left"):
        env._transfer_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._transfer_hold_counter_left

    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_transfer_hold_counter_left_last_step"):
        env._transfer_hold_counter_left_last_step = -2
    if env._transfer_hold_counter_left_last_step != step_count:
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0
        counter = torch.where(lifted_now > 0, counter + 1, torch.zeros_like(counter))
        env._transfer_hold_counter_left = counter
        env._transfer_hold_counter_left_last_step = step_count

    return (counter >= hold_steps).to(dtype=obj.data.root_pos_w.dtype)


def _transport_progress_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """DexPour ν gate for transport stage activation."""
    cfg = getattr(env, "cfg", None)
    use_soft_nu = bool(getattr(cfg, "dexpour_use_soft_nu_gate", False))
    nu = _stage_nu(env, object_cfg, eef_link_name)
    if not use_soft_nu:
        return nu
    mu = _stage_mu(env, object_cfg, eef_link_name)
    soft = _transport_soft_gate(env, object_cfg)
    return torch.maximum(nu, mu * soft)


def _left_finger_contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    contact_threshold: float = 0.02,
) -> torch.Tensor:
    """Estimate per-finger contact flags from contact sensor forces."""
    del object_cfg  # kept for backward signature compatibility
    force_magnitudes, sensor_body_names = _sensor_force_magnitudes_filtered(env, sensor_cfg)
    return _finger_contact_flags_from_sensor(force_magnitudes, contact_threshold, sensor_body_names)


def contact_finger_coverage_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    contact_threshold: float = 0.02,
    min_fingers_bonus: int = 4,
    bonus_scale: float = 1.0,
    require_thumb_contact: bool = False,
) -> torch.Tensor:
    """Reward broader multi-finger coverage to avoid 2-3-finger local optima."""
    contact_flags = _left_finger_contact_flags(
        env,
        sensor_cfg=sensor_cfg,
        object_cfg=object_cfg,
        contact_threshold=contact_threshold,
    )
    num_fingers = contact_flags.sum(dim=1).float()
    coverage = num_fingers / 5.0

    min_fingers_bonus = max(1, min(5, int(min_fingers_bonus)))
    bonus_span = float(max(1, 6 - min_fingers_bonus))
    bonus = torch.clamp((num_fingers - float(min_fingers_bonus - 1)) / bonus_span, 0.0, 1.0)

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    transfer_gate = _transport_progress_gate(env, object_cfg, eef_link_name)
    reward = lambda_trigger * (1.0 - transfer_gate) * (coverage + float(bonus_scale) * bonus)
    if require_thumb_contact:
        thumb_contact = contact_flags[:, 0].float()
        reward = reward * thumb_contact
    return reward


def strict_grasp_lift_success(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    contact_threshold: float = 0.02,
    required_fingers: int = 4,
    minimal_height: float = 0.04,
    hold_steps: int = 8,
    require_thumb_contact: bool = False,
) -> torch.Tensor:
    """Binary success metric: multi-finger grasp maintained while object is lifted."""
    obj: RigidObject = env.scene[object_cfg.name]
    contact_flags = _left_finger_contact_flags(
        env,
        sensor_cfg=sensor_cfg,
        object_cfg=object_cfg,
        contact_threshold=contact_threshold,
    )
    num_fingers = contact_flags.sum(dim=1)
    required_fingers = max(1, min(5, int(required_fingers)))
    hold_steps = max(1, int(hold_steps))

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    success_now = (num_fingers >= required_fingers) & (obj.data.root_pos_w[:, 2] > minimal_height) & (lambda_trigger > 0.5)
    if require_thumb_contact:
        success_now = success_now & contact_flags[:, 0]

    if not hasattr(env, "_strict_grasp_success_counter"):
        env._strict_grasp_success_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._strict_grasp_success_counter

    # Update once per sim step even if queried by multiple terms.
    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_strict_grasp_success_last_step"):
        env._strict_grasp_success_last_step = -2
    if env._strict_grasp_success_last_step != step_count:
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0
        counter = torch.where(success_now, counter + 1, torch.zeros_like(counter))
        env._strict_grasp_success_counter = counter
        env._strict_grasp_success_last_step = step_count

    transfer_gate = _transport_progress_gate(env, object_cfg, eef_link_name)
    return (1.0 - transfer_gate) * (counter >= hold_steps).to(dtype=obj.data.root_pos_w.dtype)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Binary reward if object is lifted above minimal height.

    Only activates when EE has reached the grasp position first.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cfg = getattr(env, "cfg", None)
    transfer_height = float(getattr(cfg, "transfer_switch_height", max(minimal_height + 1e-3, 0.10)))
    transfer_height = max(transfer_height, minimal_height + 1e-6)
    mu_trigger = _stage_mu(env, object_cfg, eef_link_name)
    nu_trigger = _transport_progress_gate(env, object_cfg, eef_link_name)
    lift_progress = torch.clamp((obj.data.root_pos_w[:, 2] - minimal_height) / (transfer_height - minimal_height), 0.0, 1.0)
    # Stage-2 transport-preparation reward: active when μ=1 and ν=0.
    return mu_trigger * (1.0 - nu_trigger) * lift_progress


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward tracking the goal pose using tanh-kernel.

    Only activates when EE has reached the grasp position first.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    nu_trigger = _transport_progress_gate(env, object_cfg, eef_link_name)
    return nu_trigger * (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_displacement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    threshold: float = 0.02,
) -> torch.Tensor:
    """Penalize object movement from initial position (XY only)."""
    obj: RigidObject = env.scene[object_cfg.name]

    current_pos = obj.data.root_pos_w[:, :2]
    initial_pos = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")

    displacement = torch.norm(current_pos - initial_pos, dim=1)
    penalty = torch.clamp(displacement - threshold, min=0.0)

    return penalty


def finger_normal_range_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize left thumb+pinky joints going outside their normal curl range.

    Returns total violation amount (positive). Use negative weight in config.
    Joints outside the normal range (e.g. bending backward) accumulate violation in radians.
    """
    robot = env.scene["robot"]

    # Left hand normal ranges (from user-confirmed curl directions)
    # 1_1 (thumb spread) excluded - full range is acceptable
    _RANGES = {
        "lj_dg_1_2": (0.0, 1.571),      # positive = curl
        "lj_dg_1_3": (-1.571, 0.0),      # negative = curl
        "lj_dg_1_4": (-1.571, 0.0),      # negative = curl
        "lj_dg_5_1": (-0.1, 0.1),        # should stay near 0
        "lj_dg_5_2": (-0.611, 0.05),     # 0.0 ideal, slight positive tolerance
        "lj_dg_5_3": (0.0, 1.571),       # positive = curl
        "lj_dg_5_4": (0.0, 1.571),       # positive = curl
    }

    total_violation = torch.zeros(env.num_envs, device=env.device)

    for joint_name, (lo, hi) in _RANGES.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_violation += torch.clamp(lo - pos, min=0.0) + torch.clamp(pos - hi, min=0.0)

    return total_violation


def finger_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward left thumb+pinky staying near initial open pose during reaching.

    Prevents excessive curling into the palm while approaching.
    Deactivates once reaching is stably complete to allow free grasping.
    """
    robot = env.scene["robot"]

    # Target = initial positions (open/ready pose)
    _TARGETS = {
        "lj_dg_1_2": 1.571,    # max open
        "lj_dg_1_3": 0.0,
        "lj_dg_1_4": 0.0,
        "lj_dg_5_3": 0.0,
        "lj_dg_5_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def thumb_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward thumb (finger 1) staying near initial open pose during reaching."""
    robot = env.scene["robot"]
    target_joints = {
        "lj_dg_1_2": 1.571,
        "lj_dg_1_3": 0.0,
        "lj_dg_1_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in target_joints.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def pinky_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward pinky (finger 5) staying near initial open pose during reaching."""
    robot = env.scene["robot"]
    target_joints = {
        "lj_dg_5_3": 0.0,
        "lj_dg_5_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in target_joints.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def finger_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward all left fingers closing toward grasp pose after reaching is complete.

    Only active after _is_reaching_stably_complete.
    Acts like a binary gripper: maximally close all fingers.
    """
    robot = env.scene["robot"]

    _CLOSE_POSE = {
        # Thumb
        "lj_dg_1_1": 0.0, "lj_dg_1_2": 1.4, "lj_dg_1_3": -0.5, "lj_dg_1_4": -0.9,
        # Index
        "lj_dg_2_1": 0.0, "lj_dg_2_2": 0.5, "lj_dg_2_3": 0.8, "lj_dg_2_4": 1.0,
        # Middle
        "lj_dg_3_1": 0.0, "lj_dg_3_2": 0.5, "lj_dg_3_3": 0.8, "lj_dg_3_4": 1.0,
        # Ring
        "lj_dg_4_1": 0.0, "lj_dg_4_2": 0.5, "lj_dg_4_3": 0.8, "lj_dg_4_4": 1.0,
        # Pinky
        "lj_dg_5_1": 0.0, "lj_dg_5_2": 0.0, "lj_dg_5_3": 0.9, "lj_dg_5_4": 0.9,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _CLOSE_POSE.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    mu_trigger = _stage_mu(env, object_cfg, eef_link_name)
    # Pre-contact closing incentive: active after approach complete, before secure grasp.
    return lambda_trigger * (1.0 - mu_trigger) * reward


def finger_wrap_cylinder_reward(
    env: ManagerBasedRLEnv,
    target_radius: float = 0.04,
    radial_std: float = 0.015,
    opposition_weight: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward cylindrical wrap around the cup in XY.

    - Radial term: fingertip ring around cup radius.
    - Opposition term: thumb opposing mean direction of other fingers.
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]

    tip_info = [
        ("thumb", "tesollo_left_ll_dg_1_4", "y", -0.0363),
        ("index", "tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("middle", "tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("ring", "tesollo_left_ll_dg_4_4", "z", 0.0255),
        ("pinky", "tesollo_left_ll_dg_5_4", "z", 0.0363),
    ]

    tip_xy_by_name: dict[str, torch.Tensor] = {}
    radial_reward_sum = torch.zeros(env.num_envs, device=env.device)
    tip_count = 0

    for finger_name, body_name, offset_axis, offset_val in tip_info:
        if body_name not in robot.data.body_names:
            continue
        body_idx = robot.data.body_names.index(body_name)
        link_pos = robot.data.body_pos_w[:, body_idx]
        link_quat = robot.data.body_quat_w[:, body_idx]

        if offset_axis == "x":
            local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device, dtype=link_pos.dtype)
        elif offset_axis == "y":
            local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device, dtype=link_pos.dtype)
        else:
            local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device, dtype=link_pos.dtype)

        world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
        tip_xy = (link_pos + world_offset)[:, :2]
        tip_xy_by_name[finger_name] = tip_xy

        radial_dist = torch.norm(tip_xy - cup_pos_xy, dim=1)
        radial_error = torch.abs(radial_dist - target_radius)
        radial_reward_sum += 1.0 - torch.tanh(radial_error / max(radial_std, 1e-6))
        tip_count += 1

    if tip_count == 0:
        return torch.zeros(env.num_envs, device=env.device)

    radial_reward_mean = radial_reward_sum / float(tip_count)
    opposition_reward = torch.zeros(env.num_envs, device=env.device)

    if "thumb" in tip_xy_by_name:
        other_vectors = [tip_xy_by_name[k] - cup_pos_xy for k in ("index", "middle", "ring", "pinky") if k in tip_xy_by_name]
        if other_vectors:
            thumb_vec = tip_xy_by_name["thumb"] - cup_pos_xy
            others_mean_vec = torch.stack(other_vectors, dim=0).mean(dim=0)
            thumb_unit = thumb_vec / (torch.norm(thumb_vec, dim=1, keepdim=True) + 1e-6)
            others_unit = others_mean_vec / (torch.norm(others_mean_vec, dim=1, keepdim=True) + 1e-6)
            opposition_reward = torch.clamp(-torch.sum(thumb_unit * others_unit, dim=1), min=0.0, max=1.0)

    opposition_weight = float(max(0.0, min(1.0, opposition_weight)))
    reward = (1.0 - opposition_weight) * radial_reward_mean + opposition_weight * opposition_reward
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * reward


def finger_wrap_coverage_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward angular spread of fingertips around cup center in XY."""
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]

    tip_info = [
        ("tesollo_left_ll_dg_1_4", "y", -0.0363),
        ("tesollo_left_ll_dg_2_4", "z", 0.0255),
        ("tesollo_left_ll_dg_3_4", "z", 0.0255),
        ("tesollo_left_ll_dg_4_4", "z", 0.0255),
        ("tesollo_left_ll_dg_5_4", "z", 0.0363),
    ]

    unit_vecs = []
    for body_name, offset_axis, offset_val in tip_info:
        if body_name not in robot.data.body_names:
            continue
        body_idx = robot.data.body_names.index(body_name)
        link_pos = robot.data.body_pos_w[:, body_idx]
        link_quat = robot.data.body_quat_w[:, body_idx]

        if offset_axis == "x":
            local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device, dtype=link_pos.dtype)
        elif offset_axis == "y":
            local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device, dtype=link_pos.dtype)
        else:
            local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device, dtype=link_pos.dtype)

        world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
        tip_xy = (link_pos + world_offset)[:, :2]
        vec = tip_xy - cup_pos_xy
        unit_vecs.append(vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-6))

    if len(unit_vecs) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    pair_scores = torch.zeros(env.num_envs, device=env.device)
    pair_count = 0
    for i in range(len(unit_vecs)):
        for j in range(i + 1, len(unit_vecs)):
            cos_ij = torch.sum(unit_vecs[i] * unit_vecs[j], dim=1).clamp(-1.0, 1.0)
            pair_scores += (1.0 - cos_ij) * 0.5
            pair_count += 1

    coverage_reward = pair_scores / float(max(1, pair_count))
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * coverage_reward


def finger_tip_orientation_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward fingertip normals pointing toward cup center (XY plane).

    Active in pre-contact grasp phase (lambda=1, mu=0) to guide cup-facing hand posture.
    """
    del std  # interface compatibility; this reward uses bounded dot-alignment directly.
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    cup_pos_xy = obj.data.root_pos_w[:, :2]

    # (body_name, offset_axis, offset_val, normal_axis)
    tip_info = [
        ("tesollo_left_ll_dg_1_4", "y", -0.0363, "y"),  # thumb
        ("tesollo_left_ll_dg_2_4", "z", 0.0255, "x"),   # index
        ("tesollo_left_ll_dg_3_4", "z", 0.0255, "x"),   # middle
        ("tesollo_left_ll_dg_4_4", "z", 0.0255, "x"),   # ring
        ("tesollo_left_ll_dg_5_4", "z", 0.0363, "x"),   # pinky
    ]

    total_reward = torch.zeros(env.num_envs, device=env.device)
    num_tips = 0

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    for body_name, offset_axis, offset_val, normal_axis in tip_info:
        if body_name not in robot.data.body_names:
            continue
        body_idx = robot.data.body_names.index(body_name)
        link_pos = robot.data.body_pos_w[:, body_idx]
        link_quat = robot.data.body_quat_w[:, body_idx]

        if offset_axis == "x":
            local_offset = torch.tensor([offset_val, 0.0, 0.0], device=env.device)
        elif offset_axis == "y":
            local_offset = torch.tensor([0.0, offset_val, 0.0], device=env.device)
        else:
            local_offset = torch.tensor([0.0, 0.0, offset_val], device=env.device)

        world_offset = quat_apply(link_quat, local_offset.unsqueeze(0).repeat(env.num_envs, 1))
        tip_pos = link_pos + world_offset
        tip_pos_xy = tip_pos[:, :2]

        if normal_axis == "x":
            local_normal = x_axis.unsqueeze(0).repeat(env.num_envs, 1)
        elif normal_axis == "y":
            local_normal = y_axis.unsqueeze(0).repeat(env.num_envs, 1)
        else:
            local_normal = z_axis.unsqueeze(0).repeat(env.num_envs, 1)

        normal_world = quat_apply(link_quat, local_normal)
        normal_xy = normal_world[:, :2]
        normal_xy = normal_xy / (torch.norm(normal_xy, dim=1, keepdim=True) + 1e-6)

        dir_to_cup = cup_pos_xy - tip_pos_xy
        dir_to_cup = dir_to_cup / (torch.norm(dir_to_cup, dim=1, keepdim=True) + 1e-6)

        alignment = torch.sum(normal_xy * dir_to_cup, dim=1)
        tip_reward = torch.clamp(alignment, min=0.0)

        total_reward += tip_reward
        num_tips += 1

    if num_tips > 0:
        total_reward = total_reward / float(num_tips)

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    mu_trigger = _stage_mu(env, object_cfg, eef_link_name)
    return lambda_trigger * (1.0 - mu_trigger) * total_reward


def contact_persistence_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contacts: int = 3,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    require_thumb_contact: bool = False,
) -> torch.Tensor:
    """Reward maintaining sufficient finger-level contacts."""
    force_magnitudes, sensor_body_names = _sensor_force_magnitudes_filtered(env, sensor_cfg)
    finger_flags = _finger_contact_flags_from_sensor(force_magnitudes, contact_threshold, sensor_body_names)
    num_contacts = finger_flags.sum(dim=-1).float()
    reward = torch.clamp(num_contacts / float(max(min_contacts, 1)), 0.0, 1.0)
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    transfer_gate = _transport_progress_gate(env, object_cfg, eef_link_name)
    reward = lambda_trigger * (1.0 - transfer_gate) * reward
    if require_thumb_contact:
        thumb_contact = finger_flags[:, 0].float()
        reward = reward * thumb_contact
    return reward


def pregrasp_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    contact_threshold: float = 0.02,
    max_allowed_contacts: int = 1,
) -> torch.Tensor:
    """Penalty for excessive finger contacts before full grasp gate opens."""
    force_magnitudes, sensor_body_names = _sensor_force_magnitudes_filtered(env, sensor_cfg)
    finger_flags = _finger_contact_flags_from_sensor(force_magnitudes, contact_threshold, sensor_body_names)
    num_contacts = finger_flags.sum(dim=-1).float()

    max_allowed_contacts = max(0, min(4, int(max_allowed_contacts)))
    excess = torch.clamp(num_contacts - float(max_allowed_contacts), min=0.0)
    excess = excess / float(max(1, 5 - max_allowed_contacts))

    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    mu_trigger = _stage_mu(env, object_cfg, eef_link_name, sensor_cfg=sensor_cfg)
    pregrasp_band = torch.clamp(lambda_trigger - mu_trigger, 0.0, 1.0)
    return pregrasp_band * excess


def synergy_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward fingers 2-4 staying near open pose during reaching."""
    robot = env.scene["robot"]
    target_joints = {
        "lj_dg_2_2": 0.0, "lj_dg_2_3": 0.0, "lj_dg_2_4": 0.0,
        "lj_dg_3_2": 0.0, "lj_dg_3_3": 0.0, "lj_dg_3_4": 0.0,
        "lj_dg_4_2": 0.0, "lj_dg_4_3": 0.0, "lj_dg_4_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in target_joints.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)
    lambda_trigger = _stage_lambda(env, object_cfg, eef_link_name)
    return (1.0 - lambda_trigger) * reward


def slip_magnitude_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    max_slip: float = 0.15,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 0.05,
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Penalty for fingertip-object relative slip."""
    robot = env.scene[robot_cfg.name]
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]

    obj = env.scene[object_cfg.name]
    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    rel_vel = link_vel - obj_vel
    slip_mag = torch.norm(rel_vel, dim=-1)
    avg_slip = slip_mag.mean(dim=-1)

    penalty = 1.0 - torch.exp(-torch.square(avg_slip / max(max_slip, 1e-6)))

    if sensor_cfg is not None:
        contact_sensor = env.scene[sensor_cfg.name]
        force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
        if sensor_cfg.body_ids is not None:
            force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]
        has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
        penalty = penalty * has_contact.float()

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * penalty


def normal_force_stability_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward smooth contact-force changes over time."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    buffer_name = f"_prev_force_mags_{sensor_cfg.name}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        delta = torch.abs(force_magnitudes - prev_forces)
        stability = 1.0 / (1.0 + delta.mean(dim=-1))
    else:
        stability = torch.ones(env.num_envs, device=env.device)
    setattr(env, buffer_name, force_magnitudes.clone())

    has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * stability * has_contact.float()


def force_spike_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    spike_threshold: float = 10.0,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Penalty for abrupt contact-force spikes."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    buffer_name = f"_prev_force_rate_{sensor_cfg.name}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        force_rate = torch.abs(force_magnitudes - prev_forces) / max(env.step_dt, 1e-6)
        max_rate = force_rate.max(dim=-1)[0]
        penalty = torch.clamp((max_rate - spike_threshold) / max(spike_threshold, 1e-6), 0.0, 1.0)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)
    setattr(env, buffer_name, force_magnitudes.clone())

    has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * penalty * has_contact.float()


def overgrip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force_range: tuple[float, float] = (1.0, 12.0),
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Penalty for under/over target grip force band."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    total_force = force_magnitudes.sum(dim=-1)
    min_force, max_force = target_force_range

    undergrip = torch.clamp(min_force - total_force, 0.0, max(min_force, 1e-6)) / max(min_force, 1e-6)
    overgrip = torch.clamp(total_force - max_force, 0.0, max(max_force, 1e-6)) / max(max_force, 1e-6)
    penalty = undergrip + overgrip

    has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * penalty * has_contact.float()
