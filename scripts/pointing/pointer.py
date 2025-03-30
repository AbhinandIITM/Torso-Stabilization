import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Screen resolution (adjust according to your screen)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam capture
cap = cv2.VideoCapture(3)
SMOOTHING_FACTOR = 2  # Higher value = smoother movement
prev_x, prev_y = 0, 0  # Store previous cursor position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.flip(frame, 1)  # Flip for natural movement
    h, w, _ = frame.shape  # Get webcam resolution

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract index finger tip (Landmark 8)
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Map coordinates to screen size
            screen_x = np.interp(x, [0, w], [0, SCREEN_WIDTH])
            screen_y = np.interp(y, [0, h], [0, SCREEN_HEIGHT])

            # Apply smoothing
            smooth_x = (prev_x * (SMOOTHING_FACTOR - 1) + screen_x) / SMOOTHING_FACTOR
            smooth_y = (prev_y * (SMOOTHING_FACTOR - 1) + screen_y) / SMOOTHING_FACTOR

            prev_x, prev_y = smooth_x, smooth_y  # Update previous values

            # Move mouse pointer
            pyautogui.moveTo(smooth_x, smooth_y)

            # Draw pointer on camera feed
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Optional: Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display video feed
    cv2.imshow("Hand Tracking Pointer", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
