import cv2
import numpy as np
import mediapipe as mp
import os

# Load the calibration data (camera matrix and distortion coefficients)
root = os.getcwd()  # Assuming the root directory is the current working directory
calib_data_path = os.path.join(root, 'charuco_calib', 'calib_data', 'MultiMatrix.npz') 
cam_mat = np.load(calib_data_path)["camMatrix"]
dist_coef = np.load(calib_data_path)["distCoef"]

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open video capture
cap = cv2.VideoCapture(2)  # Use camera 2, change if necessary

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Undistort the frame using the camera matrix and distortion coefficients
    undistorted_frame = cv2.undistort(frame, cam_mat, dist_coef)

    # Convert the image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)

    # Process the image to find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_world_landmarks:
    
            # Extract wrist coordinates (index 0 for the wrist)
            wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            wrist_z = landmarks.landmark[mp_hands.HandLandmark.WRIST].z

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = undistorted_frame.shape
            wrist_x_px = int(wrist_x * w)  # Convert normalized x to pixel x
            wrist_y_px = int(wrist_y * h)  # Convert normalized y to pixel y

            # Draw wrist point on the frame at the wrist position
            cv2.circle(undistorted_frame, (wrist_x_px, wrist_y_px), 10, (0, 255, 0), -1)

            # Draw lines connecting the wrist to other landmarks
            for i, landmark in enumerate(landmarks.landmark):
                # Skip the wrist itself (index 0)
                if i == mp_hands.HandLandmark.WRIST:
                    continue

                # Extract the other landmarks' coordinates
                x = landmark.x
                y = landmark.y
                z = landmark.z

                # Convert normalized coordinates to pixel coordinates
                x_px = int(x * w)
                y_px = int(y * h)

                # Draw line from wrist to this landmark
                cv2.line(undistorted_frame, (wrist_x_px, wrist_y_px), (x_px, y_px), (255, 0, 0), 2)  # Blue line

                # Optionally, draw the other landmark point as well
                cv2.circle(undistorted_frame, (x_px, y_px), 5, (0, 0, 255), -1)  # Red point for other landmarks

            # Display the real-world depth and coordinates near the wrist
            Z_norm = wrist_z  # Normalized depth from MediaPipe
            Z_real = (Z_norm * cam_mat[0, 0]) / wrist_x  # Real depth using focal length f_x
            cv2.putText(undistorted_frame, f"X: {wrist_x:.2f} Y: {wrist_y:.2f} Z: {Z_real:.2f}m", 
                        (wrist_x_px + 10, wrist_y_px - 10),  # Slightly offset to avoid overlap
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the undistorted frame with wrist depth and lines
    cv2.imshow("Wrist Depth with Lines", undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
