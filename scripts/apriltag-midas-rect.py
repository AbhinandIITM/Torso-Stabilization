import sys
import os

# # Get the absolute path to the `class_files/` directory
# script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to `scripts/`
# class_files_dir = os.path.join(script_dir, "../class_files")  # Path to `class_files/`

# # Add `class_files/` to Python's module search path
# sys.path.append(class_files_dir)
from class_files.Segment import Segmentation
from class_files.MiDaS_depth import MiDaS_depth
from class_files.ApriltagModule import ApriltagModule
import cv2
import numpy as np
from dt_apriltags import Detector
import time
import os
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Avoids potential conflicts with Qt
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Forces OpenCV to use XCB instead of Qt

root = os.getcwd()
calib_data_path = os.path.join(root,'charuco_calib', 'calib_data', 'MultiMatrix.npz') 
cam_mat = np.load(calib_data_path)["camMatrix"]
dist_coef = np.load(calib_data_path)["distCoef"]
segment = Segmentation()
depth = MiDaS_depth()
apriltag = ApriltagModule(calib_data_path=calib_data_path,family='tag36h11',tag_size=0.05)
print(dist_coef)

def process_video():
    cap = cv2.VideoCapture(2)
    while cap.isOpened():
        try:
            success, frame = cap.read()
            if not success:
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame_resize = cv2.resize(rgb_frame, (256, 256))
            relative_depth_map = depth.get_depthmap(rgb_frame)  
            cv2.namedWindow("Relative Depth Map", cv2.WINDOW_GUI_NORMAL)  # Set window properties
            cv2.imshow('Relative Depth Map', relative_depth_map)
            # relative_depth_map = cv2.resize(relative_depth_map, (frame.shape[1], frame.shape[0]))
            
            h, w, _ = frame.shape  # Get height and width before using them

            # ===== AprilTag Detection =====
            tags = apriltag.get_tags(frame)

            scaling_factor = 1.0
            if len(tags)>0:
                apriltag_depth, scaling_factor = apriltag.get_scaling_factor(tags=tags,frame=frame,relative_depth_map=relative_depth_map)
                if apriltag_depth:
                    cv2.putText(frame, f"Tag Depth: {round(apriltag_depth * 100, 2)} cm", (20, 30),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

                x,y = segment.get_smoothed_tip(rgb_frame)
                if x is not None and y is not None:
                    roi_size = 100
                    x1, y1 = max(0, int(x - (roi_size / 2))), max(0, int(y - (roi_size / 2)))
                    x2, y2 = min(w, int(x + (roi_size / 2))), min(h, int(y + (roi_size / 2)))
                    roi = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    #Canny edge detection
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cv2.drawContours(frame[y1:y2, x1:x2], [cnt], -1, (0, 255, 0), 2)

                    rect_relative_depth = np.mean(relative_depth_map[y1:y2, x1:x2])
                    rect_absolute_depth = rect_relative_depth * scaling_factor if scaling_factor else None
                    cv2.putText(frame, f"Rect Depth: {round(rect_absolute_depth * 100, 2) if rect_absolute_depth else 'N/A'} cm",
                                 (20, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                    
                    if apriltag_depth is not None and rect_absolute_depth is not None:
                        # Convert 2D image point to normalized camera coordinates
                        rect_center_distorted = np.array([[x, y]], dtype=np.float32)  # Ensure it's a 2D array
                        rect_center = cv2.undistortPoints(rect_center_distorted, cam_mat, dist_coef)

                        # Convert to homogeneous coordinates (normalized camera space)
                        rect_center_homogeneous = np.array([rect_center[0][0][0], -rect_center[0][0][1], 1.0])

                        # Scale with depth to get real-world 3D coordinates
                        rect_center_3d = rect_center_homogeneous * rect_absolute_depth

                        # Display coordinates relative to AprilTag
                        cv2.putText(frame, f"X: {rect_center_3d[0]:.2f}, Y: {rect_center_3d[1]:.2f}, Z: {rect_center_3d[2]:.2f}",
                                    (20, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

                        # === Plot optical center ===
                        
                        optical_center = (int(cam_mat[0, 2]), int(cam_mat[1, 2]))  # (cx, cy)
                        cv2.circle(frame, optical_center, 5, (0, 0, 255), -1)  # Red dot for optical center

            yield frame

        except Exception as e:
            print(e)
            break
    cap.release()
    cv2.destroyAllWindows()



for frame in process_video():
    
    cv2.namedWindow("AprilTag & Hand Depth", cv2.WINDOW_GUI_NORMAL)  # Set window properties
    cv2.imshow("AprilTag & Hand Depth", frame)  # Show frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
