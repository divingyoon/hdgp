import omni.usd
from pxr import Gf, UsdPhysics
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
    # 1. Stage의 모든 프림을 순회합니다.
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        
        # 2. 이름에 "sensor_link"가 포함된 프림만 필터링합니다. (존재하는 것만 대상)
        if "sensor_link" in prim_name:
            # 3. 이미 센서가 달려있는지 확인 (중복 생성 방지)
            sensor_path = f"{prim_path}/Contact_Sensor"
            if stage.GetPrimAtPath(sensor_path):
                print(f"[Skip] Sensor already exists on {prim_name}")
                continue
            
            try:
                # 4. 존재하는 링크 바로 아래에 센서 생성
                ContactSensor(
                    prim_path=sensor_path,
                    name=f"{prim_name}_contact_sensor",
                    radius=0.005, # 1cm 지름
                    min_threshold=0.0,
                    max_threshold=1.0e6,
                )
                print(f"[Done] Created sensor on: {prim_path}")
                created_count += 1
            except Exception as e:
                print(f"[Error] Failed on {prim_name}: {e}")

    print(f"--- Finished! Total {created_count} sensors created on existing links. ---")

if __name__ == "__main__":
    main()
