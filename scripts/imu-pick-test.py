import cv2
import numpy as np
import os
from ultralytics import YOLO
import logging
from utils.MiDaS_depth import MiDaS_depth
from utils.ApriltagModule import ApriltagModule
from utils.imu_utils import IMU
from scipy.spatial.transform import Rotation as R

# ---------- Load Camera Calibration ----------
root = os.getcwd()
calib_data_path = os.path.join(root, 'charuco_calib', 'calib_data', 'MultiMatrix.npz') 
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
fx, fy = cam_mat[0, 0], cam_mat[1, 1]
cx, cy = cam_mat[0, 2], cam_mat[1, 2]

# ---------- Setup ----------
model = YOLO("yolo11n-seg.pt")
logging.getLogger('ultralytics').setLevel(logging.ERROR)
depth = MiDaS_depth()
apriltag = ApriltagModule(calib_data_path=calib_data_path, family='tag36h11', tag_size=0.05)
imu = IMU()

COCO_CLASSES = model.names
BOTTLE_CLASS_ID = [i for i, name in COCO_CLASSES.items() if name == "bottle"]
if not BOTTLE_CLASS_ID:
    raise ValueError("Bottle class not found.")
BOTTLE_CLASS_ID = BOTTLE_CLASS_ID[0]

stored_world_pos = None

# ---------- Main Loop ----------
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map = depth.get_depthmap(rgb_frame)
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    tags = apriltag.get_tags(frame)
    scaling_factor = 1.0
    if tags:
        _, scaling_factor = apriltag.get_scaling_factor(tags=tags, frame=frame, relative_depth_map=depth_map)
        depth_map *= scaling_factor

    results = model(frame)[0]
    bottle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if results.masks is not None:
        for seg, cls in zip(results.masks.xy, results.boxes.cls):
            if int(cls) == BOTTLE_CLASS_ID:
                points = np.array(seg, dtype=np.int32)
                cv2.fillPoly(bottle_mask, [points], 255)

    contours, _ = cv2.findContours(bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_3d_pos = None
    current_world_pos = None

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cxp = np.clip(x + w // 2, 0, frame.shape[1] - 1)
        cyp = np.clip(y + h, 0, frame.shape[0] - 1)

        cv2.circle(frame, (cxp, cyp), 5, (0, 255, 0), -1)
        cv2.arrowedLine(frame, (cxp, cyp), (cxp, min(cyp + 30, frame.shape[0] - 1)), (255, 0, 0), 2)

        z = depth_map[cyp, cxp]
        if z > 0:
            x3d = (cxp - cx) * z / fx
            y3d = (cyp - cy) * z / fy
            current_3d_pos = np.array([x3d, y3d, z])

            tf_data = imu.get_tf_data()
            if tf_data:
                t = tf_data["transforms"][0]["transform"]
                translation = np.array([t["translation"]["x"], t["translation"]["y"], t["translation"]["z"]])
                rotation = R.from_quat([
                    t["rotation"]["x"],
                    t["rotation"]["y"],
                    t["rotation"]["z"],
                    t["rotation"]["w"]
                ])
                T_wc = np.eye(4)
                T_wc[:3, :3] = rotation.as_matrix()
                T_wc[:3, 3] = translation

                cam_point = np.hstack([current_3d_pos, 1.0])
                current_world_pos = (T_wc @ cam_point)[:3]

    # ---------- Compare to Stored ----------
    if stored_world_pos is not None and current_world_pos is not None:
        dist = np.linalg.norm(current_world_pos - stored_world_pos)
        cv2.putText(frame, f"Lifted: {dist:.3f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Scaling factor: {scaling_factor:.3f} m", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # ---------- Overlay Mask ----------
    overlay = frame.copy()
    overlay[bottle_mask == 255] = (0, 0, 255)
    output = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # ---------- Display ----------
    cv2.imshow("Bottle Lift Detection", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32 and current_world_pos is not None:
        stored_world_pos = current_world_pos.copy()
        print("Saved WORLD 3D position:", stored_world_pos)

cap.release()
cv2.destroyAllWindows()
