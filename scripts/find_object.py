
import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import FastSAM
import sys
import os
import sys
import os
import ultralytics
cv2.ocl.setUseOpenCL(False)
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
from class_files.Segment import Segmentation
from class_files.MiDaS_depth import MiDaS_depth
from class_files.ApriltagModule import ApriltagModule

from class_files.Segment import Segmentation
from class_files.MiDaS_depth import MiDaS_depth
from class_files.ApriltagModule import ApriltagModule

seg_model = FastSAM("FastSAM-s.pt")
root = os.getcwd()
calib_data_path = os.path.join(root, 'charuco_calib', 'calib_data', 'MultiMatrix.npz') 

segment = Segmentation()
depth = MiDaS_depth()
apriltag = ApriltagModule(calib_data_path=calib_data_path,family='tag36h11',tag_size=0.05)

ROI_SIZE = 100
cap = cv2.VideoCapture(2)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    x,y  = segment.get_smoothed_tip(rgb_frame)
    h, w, _ = frame.shape
    if  x is not None and y is not None:
        canny_frame = segment.draw_canny(center=(x,y),frame=frame,roi_size=ROI_SIZE)

        blurred_frame = cv2.GaussianBlur(canny_frame, (5, 5), 0)  # Apply mild blur
        seg_frame = seg_model.predict(blurred_frame,points=[x,y])[0]
        seg_frame_plot = cv2.resize(seg_frame.plot(conf=False,labels=False),(1920,1080))
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
                    dist_sq = ((x - center_x) ** 2 + (y - center_y) ** 2)/(w*h)
                    distances.append(-dist_sq)  # Use negative squared distance for softmax
                else:
                    distances.append(-1e9)  # Very low score for out-of-ROI objects
            
            scores = np.exp(distances) / np.sum(np.exp(distances))
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, box)
                score_text = f"{scores[i]:.3f}"

                # Draw bounding box
                cv2.rectangle(seg_frame_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw score near the bounding box
                cv2.putText(seg_frame_plot, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Segmented Object", seg_frame_plot)  
        cv2.namedWindow("Segmented Object",cv2.WINDOW_NORMAL)
    else:            
        cv2.namedWindow("orig_frame",cv2.WINDOW_NORMAL)
        cv2.imshow("orig frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
