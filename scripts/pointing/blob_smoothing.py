import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Set maximum limits for the ellipse's dimensions and area
MAX_LENGTH = 200  # Maximum length (major axis) in pixels
MAX_BREADTH = 150  # Maximum breadth (minor axis) in pixels
MAX_AREA = 2000000  # Maximum area of the ellipse in square pixels

# Smoothing factor for fingertip positions (0 to 1)
SMOOTHING_FACTOR = 0.3
last_positions = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            # Define the landmarks for each finger (penultimate joint to fingertip)
            finger_indices = [
                [1, 4],    # Thumb: CMC -> Thumb tip
                [5, 8],    # Index: MCP -> Index tip
                [9, 12],   # Middle: MCP -> Middle tip
                [13, 16],  # Ring: MCP -> Ring tip
                [17, 20]   # Pinky: MCP -> Pinky tip
            ]
            
            # Store the fingertip positions and the extended points
            points = []
            for finger in finger_indices:
                start_idx, end_idx = finger
                start = np.array([hand_landmarks.landmark[start_idx].x, hand_landmarks.landmark[start_idx].y])
                end = np.array([hand_landmarks.landmark[end_idx].x, hand_landmarks.landmark[end_idx].y])
                
                # Compute direction vector from the joint to the fingertip
                vector = end - start
                
                # Normalize the vector
                vector /= np.linalg.norm(vector)
                
                # Extend the fingertip position along the direction vector
                extended_tip = end + vector * 0.1  # Extend by a factor of 0.1 for visualization

                # Convert to pixel coordinates
                extended_tip_pixel = (int(extended_tip[0] * w), int(extended_tip[1] * h))

                # Apply smoothing to the fingertip positions
                if start_idx not in last_positions:
                    last_positions[start_idx] = extended_tip_pixel
                smoothed_tip = (
                    int(last_positions[start_idx][0] * (1 - SMOOTHING_FACTOR) + extended_tip_pixel[0] * SMOOTHING_FACTOR),
                    int(last_positions[start_idx][1] * (1 - SMOOTHING_FACTOR) + extended_tip_pixel[1] * SMOOTHING_FACTOR)
                )
                last_positions[start_idx] = smoothed_tip

                points.append(smoothed_tip)

            # Create an ellipse to fit the points
            if len(points) >= 5:  # Need at least 5 points to fit an ellipse
                points = np.array(points)
                ellipse = cv2.fitEllipse(points)  # Fit ellipse to the points

                # Get the length and breadth of the ellipse
                (center, axes, angle) = ellipse
                length, breadth = axes

                # Check if length, breadth, or area exceeds the limits
                area = np.pi * (length / 2) * (breadth / 2)  # Area of the ellipse
                if length > MAX_LENGTH:
                    scale_factor = MAX_LENGTH / length
                    length = MAX_LENGTH
                    breadth *= scale_factor  # Scale the breadth to maintain aspect ratio
                if breadth > MAX_BREADTH:
                    scale_factor = MAX_BREADTH / breadth
                    breadth = MAX_BREADTH
                    length *= scale_factor  # Scale the length to maintain aspect ratio
                if area > MAX_AREA:
                    scale_factor = np.sqrt(MAX_AREA / area)
                    length *= scale_factor
                    breadth *= scale_factor

                # Update the ellipse with the scaled dimensions
                ellipse = ((center[0], center[1]), (length, breadth), angle)

                # Draw the filled ellipse
                cv2.ellipse(frame, ellipse, (0, 255, 0), -1)  # Fill the ellipse with green color

    cv2.imshow("Pointing Area Region with Scaled Ellipse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
