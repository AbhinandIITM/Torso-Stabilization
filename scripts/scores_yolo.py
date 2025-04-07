import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import logging

cv2.ocl.setUseOpenCL(False)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from class_files.Segment import Segmentation
from class_files.MiDaS_depth import MiDaS_depth
from class_files.ApriltagModule import ApriltagModule

model = YOLO('yolov8n-seg.pt')  # Using YOLOv8 segmentation model

root = os.getcwd()
calib_data_path = os.path.join(root, 'Torso-Stabilization','charuco_calib', 'calib_data', 'MultiMatrix.npz') 

segment = Segmentation()
depth = MiDaS_depth()
apriltag = ApriltagModule(calib_data_path=calib_data_path, family='tag36h11', tag_size=0.05)

cap = cv2.VideoCapture(2)
cv2.startWindowThread()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y = segment.get_smoothed_tip(rgb_frame)
    h, w, _ = frame.shape

    if x is not None and y is not None:
        results = model(frame, verbose=False)[0]
        seg_frame_plot = frame.copy()

        if results.boxes is not None:
            bboxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            class_names = model.names

            all_boxes = []
            all_classes = []
            distances = []

            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                dist_sq = ((x - center_x) ** 2 + (y - center_y) ** 2) / (w * h)
                all_boxes.append((x1, y1, x2, y2))
                all_classes.append(classes[i])
                distances.append(-dist_sq)  # Negative for softmax-based proximity

            if distances:
                scores = np.exp(distances) / np.sum(np.exp(distances))
                distances  = distances*10
                # scores = 1/(1 + np.exp(distances))  # Sigmoid-like function for proximity
                for i, (x1, y1, x2, y2) in enumerate(all_boxes):
                    label = class_names[all_classes[i]]
                    score_text = f"{label}: {scores[i]:.2f}"

                    # Draw bounding box
                    cv2.rectangle(seg_frame_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label + score above box
                    cv2.putText(seg_frame_plot, score_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        # Mark the fingertip position
        cv2.circle(seg_frame_plot, (x, y), 5, (0, 255, 255), -1)

        cv2.imshow("YOLO Segmentation - Top Prediction", seg_frame_plot)

    else:
        cv2.namedWindow("orig frame", cv2.WINDOW_NORMAL)
        cv2.imshow("orig frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
