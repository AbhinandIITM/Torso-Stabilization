
import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import FastSAM
from ultralytics import YOLO
import os
import sys
import ultralytics

cv2.ocl.setUseOpenCL(False)
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


from utils.Segment import Segmentation
from utils.MiDaS_depth import MiDaS_depth
from utils.ApriltagModule import ApriltagModule

from utils.Segment import Segmentation
from utils.MiDaS_depth import MiDaS_depth
from utils.ApriltagModule import ApriltagModule

seg_model = FastSAM("FastSAM-s.pt")
model = YOLO('yolov8n-seg.pt')
root = os.getcwd()
calib_data_path = os.path.join(root,'charuco_calib', 'calib_data', 'MultiMatrix.npz') 

segment = Segmentation()
depth = MiDaS_depth()
apriltag = ApriltagModule(calib_data_path=calib_data_path,family='tag36h11',tag_size=0.05)

ROI_SIZE = 100
cap = cv2.VideoCapture(2)
cv2.startWindowThread()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    depth_map = depth.get_depthmap(rgb_frame)  # (h, w) float32 array

    x,y  = segment.get_smoothed_tip(rgb_frame)
    h, w, _ = frame.shape
    if  x is not None and y is not None:
        canny_frame = segment.draw_canny(center=(x,y),frame=frame,roi_size=ROI_SIZE)

        blurred_frame = cv2.GaussianBlur(canny_frame, (5, 5), 0)  # Apply mild blur
        seg_frame = seg_model.predict(blurred_frame,points=[x,y])[0]
        #seg_frame_plot = cv2.resize(seg_frame.plot(conf=False,labels=True),(1920,1080))
        bboxes = seg_frame.boxes.xyxy.cpu().numpy()
        if len(bboxes) > 0:
            distances = []
            centers = []

            for box in bboxes:
                x1, y1, x2, y2 = box
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                centers.append((center_x, center_y))

                # Ensure bounding box is inside the ROI
                if x1 < x < x2 and y1 < y< y2:
                    fingertip_depth = depth_map[y, x]
                    object_depth = depth_map[center_y, center_x]

                    # 3D distance in pseudo-metric space (x, y in pixels, z from MiDaS)
                    dist_sq = ((x - center_x) ** 2 + (y - center_y) ** 2 + (fingertip_depth - object_depth) ** 2)

                    distances.append(-dist_sq)  # Use negative squared distance for softmax
                else:
                    distances.append(-1e9)  # Very low score for out-of-ROI objects
            
            scores = np.exp(distances) / np.sum(np.exp(distances))
            # Get original-size frame from seg_model
            seg_frame_plot = seg_frame.plot(conf=False, labels=False).copy()

            # Draw on this frame directly
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, box)
                score_text = f"{scores[i]:.3f}"

                cv2.rectangle(seg_frame_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(seg_frame_plot, score_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Resize for display only at the end
            seg_frame_plot_resized = cv2.resize(seg_frame_plot, (1920, 1080))
            cv2.imshow("Segmented Object", seg_frame_plot_resized)

    else:            
        cv2.namedWindow("orig frame", cv2.WINDOW_NORMAL)
        cv2.imshow("orig frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
