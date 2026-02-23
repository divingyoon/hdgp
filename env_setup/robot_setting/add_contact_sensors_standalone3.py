import omni.usd
from pxr import UsdPhysics, UsdGeom
from omni.isaac.sensor import ContactSensor
from omni.isaac.core.utils.stage import get_current_stage

def setup_physics_scene():
    stage = get_current_stage()
    scene_path = "/World/physicsScene"
    if not stage.GetPrimAtPath(scene_path):
        UsdPhysics.Scene.Define(stage, scene_path)

def main():
    stage = get_current_stage()
    setup_physics_scene()
    
    created_count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        
        if "sensor_link" in prim_name:
            # 1. 부모 프림에 Collision API가 있는지 확인하고, 없으면 강제로 추가
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                print(f"[Fixed] Applied CollisionAPI to {prim_name}")
            
            sensor_path = f"{prim_path}/Contact_Sensor"
            if stage.GetPrimAtPath(sensor_path):
                continue
            
            try:
                # 2. 이제 센서 생성 (Collision API가 적용되었으므로 에러가 나지 않음)
                ContactSensor(
                    prim_path=sensor_path,
                    name=f"{prim_name}_contact_sensor",
                    radius=0.005,
                    min_threshold=0.0,
                    max_threshold=1.0e6,
                )
                print(f"[Done] Created sensor on: {prim_path}")
                created_count += 1
            except Exception as e:
                print(f"[Error] Failed on {prim_name}: {e}")

    print(f"--- Finished! Total {created_count} sensors processed. ---")

if __name__ == "__main__":
    main()
