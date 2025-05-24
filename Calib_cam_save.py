import os
import json
import pyzed.sl as sl

# 출력 폴더
output_dir = "./Calib_cam"
os.makedirs(output_dir, exist_ok=True)

# 시리얼 ↔ 위치 매핑
camera_list = {
    41182735: "front",
    49429257: "right",
    44377151: "left",
    49045152: "top"
}

for serial, position in camera_list.items():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.set_from_serial_number(serial)
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print(f"[{position}] ZED 초기화 실패 (serial: {serial})")
        continue

    calib = zed.get_camera_information().camera_configuration.calibration_parameters

    for side_attr, side_name in [("left_cam", "leftcam"), ("right_cam", "rightcam")]:
        cam = getattr(calib, side_attr)
        # 카메라 행렬
        cam_matrix = [
            [float(cam.fx), 0.0,           float(cam.cx)],
            [0.0,           float(cam.fy), float(cam.cy)],
            [0.0,           0.0,           1.0]
        ]
        # 왜곡 계수 전체를 리스트로 변환 (필요한 만큼 모두 기록)
        dist_coeffs = [float(d) for d in cam.disto]

        cam_spec = {
            "camera_matrix": cam_matrix,
            "distortion_coeffs": dist_coeffs
        }

        filename = f"{position}_{serial}_{side_name}_calib.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            # indent=4로 저장해도 소수점은 사라지지 않습니다.
            json.dump(cam_spec, f, indent=4)
        print(f"[{position}] Saved {filename}  (distortion: {dist_coeffs})")

    zed.close()
