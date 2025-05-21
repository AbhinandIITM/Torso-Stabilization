from ultralytics import YOLO
import cv2, numpy as np, os, logging, time
from collections import defaultdict

# Optional: your own Segmentation and MiDaS depth modules
from class_files.Segment import Segmentation
from class_files.MiDaS_depth import MiDaS_depth

cv2.ocl.setUseOpenCL(False)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# === Load modules ===
segment = Segmentation()
depth = MiDaS_depth()
model = YOLO("yolov8n-seg.pt")

# === Camera ===
cap = cv2.VideoCapture(2)
cv2.startWindowThread()

# === Timings & Bottle state ===
timings = defaultdict(list)
bottle_state = {}

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

    # === Step: YOLO segmentation ===
    start = time.time()
    result = model(frame, verbose=False)[0]
    seg_frame_plot = result.plot(conf=False)
    timings['yolo_inference'].append(time.time() - start)

    # === Step: Process masks ===
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

            top_y = np.min(ys)
            bottom_y = np.max(ys)
            height = bottom_y - top_y

            if name not in bottle_state:
                bottle_state[name] = {
                    "initial_height": height,
                    "intersecting": True,
                }

            # If current height increases 1.5x → picked up
            if height > 1.5 * bottle_state[name]["initial_height"]:
                bottle_state[name]["intersecting"] = False

            intersection = bottle_state[name]["intersecting"]

            label = f"{name}: {'Touching table' if intersection else 'Picked up'}"
            color = (0, 255, 0) if intersection else (0, 0, 255)
            box = result.boxes.xyxy[i].int().cpu().numpy()
            cv2.putText(seg_frame_plot, label, (10, 30 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.rectangle(seg_frame_plot, tuple(box[:2]), tuple(box[2:]), color, 2)

    cv2.imshow("Bottle Pickup Detection", seg_frame_plot)
    timings['total_frame'].append(time.time() - start_total)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Print average timings ===
print("\n=== Average Timings (in seconds) ===")
for step, times in timings.items():
    if times:
        print(f"{step}: {np.mean(times):.3f}")

cap.release()
cv2.destroyAllWindows()
