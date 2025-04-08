from class_files.Segment import Segmentation
from class_files.MiDaS_depth import MiDaS_depth
from class_files.ApriltagModule import ApriltagModule
from ultralytics import YOLO
import cv2, numpy as np, os, logging, time
from collections import defaultdict
from class_files.Zoe_Depth import Zoe_Depth

cv2.ocl.setUseOpenCL(False)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# === Load camera calibration ===
root = os.getcwd()
calib_path = os.path.join(root, 'charuco_calib', 'calib_data', 'MultiMatrix.npz') 
calib = np.load(calib_path)
cam_mat = calib["camMatrix"]
dist_coef = calib["distCoef"]

# === Modules ===
segment = Segmentation()
depth = MiDaS_depth()
# depth = Zoe_Depth()
apriltag = ApriltagModule(calib_data_path=calib_path, family='tag36h11', tag_size=0.05)
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(2)
cv2.startWindowThread()

timings = defaultdict(list)

while cap.isOpened():
    start_total = time.time()

    success, frame = cap.read()
    if not success:
        continue

    # === Step: Preprocessing and depth ===
    start = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_small = cv2.resize(rgb, (256, 256))
    rel_depth = depth.get_depthmap(rgb_small)
    rel_depth = cv2.resize(rel_depth, (frame.shape[1], frame.shape[0]))
    timings['depth_estimation'].append(time.time() - start)

    # === Step: AprilTag detection ===
    start = time.time()
    tags = apriltag.get_tags(frame)
    timings['apriltag_detection'].append(time.time() - start)

    if tags:
        # === Step: Scaling depth ===
        start = time.time()
        abs_tag_depth, scale = apriltag.get_scaling_factor(frame, tags, rel_depth)
        abs_depth = rel_depth * scale if scale else rel_depth
        timings['scale_depth'].append(time.time() - start)

        # === Step: YOLO segmentation ===
        start = time.time()
        result = model(frame, verbose=False)[0]
        seg_frame_plot = result.plot(conf=False)
        timings['yolo_inference'].append(time.time() - start)

        # === Step: Plane estimation ===
        start = time.time()
        # normals = []
        # centers = []
        # for tag in tags:
        #     if not hasattr(tag, "pose_R") or not hasattr(tag, "pose_t"):
        #         continue
        #     R = tag.pose_R
        #     t = tag.pose_t.flatten()
        #     normal = (R @ np.array([[0], [0], [1]])).flatten()
        #     normals.append(normal)
        #     centers.append(t)

        # if len(normals) == 0:
        #     continue

        # plane_normal = np.mean(normals, axis=0)
        # plane_normal /= np.linalg.norm(plane_normal)
        # point_on_plane = np.mean(centers, axis=0)
        tag = tags[0]
        R = tag.pose_R
        t = tag.pose_t
        plane_normal = R @ np.array([[0], [0], [1]])  # Z axis
        plane_normal = plane_normal.flatten()
        point_on_plane = t.flatten()

        timings['plane_estimation'].append(time.time() - start)

        # === Step: Draw plane ===
        start = time.time()
        plane_size = 0.4
        half = plane_size / 2.0
        plane_corners_3d = np.float32([
            [-half, -half, 0],
            [ half, -half, 0],
            [ half,  half, 0],
            [-half,  half, 0]
        ])
        plane_corners_2d, _ = cv2.projectPoints(plane_corners_3d, cv2.Rodrigues(R)[0], t, cam_mat, dist_coef)
        plane_corners_2d = plane_corners_2d.astype(int).reshape(-1, 2)
        cv2.polylines(seg_frame_plot, [plane_corners_2d], isClosed=True, color=(255, 0, 255), thickness=2)
        cv2.putText(seg_frame_plot, "Tag Plane", tuple(plane_corners_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        timings['draw_plane'].append(time.time() - start)

        # === Step: Object mask processing and intersection check ===
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                cls = int(result.boxes.cls[i].item())
                name = model.names[cls]
                if name.lower() not in ["bottle", "vase"]:
                    continue


                binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                ys, xs = np.where(binary_mask == 1)
                if len(xs) < 10:
                    continue

                intersection = True
                for px, py in zip(xs, ys):
                    d = abs_depth[py, px]
                    if d == 0: continue
                    undistorted = cv2.undistortPoints(np.array([[[px, py]]], dtype=np.float32), cam_mat, dist_coef)
                    norm_dir = np.array([undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0])
                    world_pt = norm_dir * d
                    dot = np.dot(plane_normal, world_pt - point_on_plane)
                    if dot <-0.65:
                        intersection = False
                        break
                    
                label = f"{name}: {'Touching table' if intersection else 'Not touching'}  {dot:.2f}"
                color = (0, 255, 0) if intersection else (0, 0, 255)
                box = result.boxes.xyxy[i].int().cpu().numpy()
                cv2.putText(seg_frame_plot, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.rectangle(seg_frame_plot, tuple(box[:2]), tuple(box[2:]), color, 2)

        cv2.imshow("Bottle-Table Intersection", seg_frame_plot)

    else:
        cv2.imshow("Waiting for AprilTag", frame)

    timings['total_frame'].append(time.time() - start_total)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Print average timings after quitting ===
print("\n=== Average Timings (in seconds) ===")
for step, times in timings.items():
    if times:
        print(f"{step}: {np.mean(times):.3f}")

cap.release()
cv2.destroyAllWindows()
