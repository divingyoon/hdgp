from omni.isaac.kit import SimulationApp

# Configure before running.
USD_PATH = ""  # Set to a USD file path to open, or leave empty to use current stage.
HEADLESS = False

DEFAULT_OFFSET = (0.0, 0.0, 0.0)  # xyz offset for all sensors (meters)
OFFSETS = {
    # Per-link overrides, e.g.:
    # "tesollo_left_palm_sensor_link": (0.0, 0.0, 0.001),
}

DEFAULT_RADIUS = 0.005
RADIUS_OVERRIDES = {
    # Per-link overrides, e.g.:
    # "tesollo_left_palm_sensor_link": 0.01,
}

PRINT_FORCE_TORQUE = True
PRINT_EVERY = 10
NUM_STEPS = 120
ONLY_NONZERO = True


simulation_app = SimulationApp({"headless": HEADLESS})

import omni.usd  # noqa: E402
from pxr import Gf  # noqa: E402
from omni.isaac.sensor import ContactSensor  # noqa: E402
from omni.isaac.core import SimulationContext  # noqa: E402


def iter_sensor_link_names():
    names = [
        "tesollo_left_palm_sensor_link",
        "tesollo_right_palm_sensor_link",
    ]
    for side in ("left", "right"):
        for finger in range(1, 6):
            for seg in range(1, 5):
                names.append(f"tesollo_{side}_finger_{finger}_seg{seg}_sensor_link")
            names.append(f"tesollo_{side}_finger_{finger}_tip_sensor_link")
    return names


def find_prim_by_name(stage, name):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def _vec3_norm(val):
    try:
        x, y, z = float(val[0]), float(val[1]), float(val[2])
    except Exception:
        return None
    return (x * x + y * y + z * z) ** 0.5


def _extract_force_torque(sensor):
    force = None
    torque = None
    data = None
    if hasattr(sensor, "get_current_frame"):
        data = sensor.get_current_frame()
    if isinstance(data, dict):
        force = data.get("force") or data.get("forces")
        torque = data.get("torque") or data.get("torques")
    if force is None and hasattr(sensor, "get_net_force"):
        try:
            force = sensor.get_net_force()
        except Exception:
            force = None
    if torque is None and hasattr(sensor, "get_net_torque"):
        try:
            torque = sensor.get_net_torque()
        except Exception:
            torque = None
    return force, torque


def main():
    if USD_PATH:
        omni.usd.get_context().open_stage(USD_PATH)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage. Set USD_PATH or open a stage first.")

    created = 0
    sensors = []
    for link_name in iter_sensor_link_names():
        prim = find_prim_by_name(stage, link_name)
        if prim is None:
            print(f"[WARN] Missing prim for {link_name}")
            continue

        offset = OFFSETS.get(link_name, DEFAULT_OFFSET)
        radius = RADIUS_OVERRIDES.get(link_name, DEFAULT_RADIUS)
        sensor_path = f"{prim.GetPath()}/contact_sensor"

        sensor = ContactSensor(
            prim_path=sensor_path,
            name=f"{link_name}_contact_sensor",
            translation=Gf.Vec3f(*offset),
            radius=radius,
            min_threshold=0.0,
            max_threshold=1.0e6,
        )
        sensor.initialize()
        sensors.append((link_name, sensor))
        created += 1

    print(f"Contact sensors created: {created}")

    if PRINT_FORCE_TORQUE and sensors:
        simulation_context = SimulationContext(stage_units_in_meters=1.0)
        simulation_context.initialize_physics()
        simulation_context.play()
        for step in range(NUM_STEPS):
            simulation_context.step(render=not HEADLESS)
            if step % PRINT_EVERY != 0:
                continue
            for link_name, sensor in sensors:
                force, torque = _extract_force_torque(sensor)
                if ONLY_NONZERO:
                    f_norm = _vec3_norm(force) if force is not None else None
                    t_norm = _vec3_norm(torque) if torque is not None else None
                    if (f_norm is None or f_norm == 0.0) and (t_norm is None or t_norm == 0.0):
                        continue
                print(f\"[{step:04d}] {link_name} force={force} torque={torque}\")
        simulation_context.stop()

    stage.GetRootLayer().Save()


if __name__ == "__main__":
    main()
    simulation_app.close()
