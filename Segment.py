import mediapipe as mp
import numpy as np
import cv2

class Segmentation:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.SMOOTHING_FACTOR = 0.3
        self.last_positions = {}

    def get_smoothed_tip(self, frame):
        rgb_frame = frame
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape

                # Index fingertip (landmark 8) and base (landmark 5)
                start_idx, end_idx = 5, 8
                start = np.array([hand_landmarks.landmark[start_idx].x, hand_landmarks.landmark[start_idx].y])
                end = np.array([hand_landmarks.landmark[end_idx].x, hand_landmarks.landmark[end_idx].y])

                # Compute direction vector and extend fingertip position
                vector = end - start
                norm = np.linalg.norm(vector)
                if norm == 0:
                    return None, None  # Avoid division by zero

                vector /= norm
                extended_tip = end + vector * 0.1  # Extend by a small factor

                # Convert to pixel coordinates
                extended_tip_pixel = (int(extended_tip[0] * w), int(extended_tip[1] * h))

                # Smoothing process
                if end_idx not in self.last_positions:
                    self.last_positions[end_idx] = extended_tip_pixel
                
                smoothed_tip = (
                    int(self.last_positions[end_idx][0] * (1 - self.SMOOTHING_FACTOR) + extended_tip_pixel[0] * self.SMOOTHING_FACTOR),
                    int(self.last_positions[end_idx][1] * (1 - self.SMOOTHING_FACTOR) + extended_tip_pixel[1] * self.SMOOTHING_FACTOR)
                )

                # Ensure coordinates remain within frame bounds
                x = max(0, min(smoothed_tip[0], w - 1))
                y = max(0, min(smoothed_tip[1], h - 1))
                self.last_positions[end_idx] = (x, y)

                return x, y

        return None, None


    # def calculate_roi_center(self,hand_landmarks, frame_width, frame_height):
    #     WEIGHTS = np.array([0.15, 0.15, 0.075, 0.075, 0.075, 0.075])  # Thumb, Index, Middle, Ring, Pinky, Palm center
    #     EXTENSION_FACTOR = 2  # Factor to extend the ROI center outward
        
    #     wrist = np.array([hand_landmarks[0].x * frame_width, hand_landmarks[0].y * frame_height])

    #     # Fingertip positions
    #     tips = [
    #         np.array([hand_landmarks[4].x * frame_width, hand_landmarks[4].y * frame_height]),   # Thumb tip
    #         np.array([hand_landmarks[8].x * frame_width, hand_landmarks[8].y * frame_height]),   # Index finger tip
    #         np.array([hand_landmarks[12].x * frame_width, hand_landmarks[12].y * frame_height]), # Middle finger tip
    #         np.array([hand_landmarks[16].x * frame_width, hand_landmarks[16].y * frame_height]), # Ring finger tip
    #         np.array([hand_landmarks[20].x * frame_width, hand_landmarks[20].y * frame_height])  # Pinky tip
    #     ]

    #     # Compute palm center (midpoint between the wrist and fingertips)
    #     palm_center = np.mean(tips, axis=0)
    #     tips.append(palm_center)

    #     # Compute vectors from wrist to each tip
    #     vectors = [tip - wrist for tip in tips]

    #     # Weighted sum of vectors
    #     weighted_vector = sum(w * v for w, v in zip(WEIGHTS, vectors))

    #     # Compute initial ROI center
    #     roi_center = wrist + weighted_vector

    #     # === Extension from palm center ===
    #     extension_vector = roi_center - wrist
    #     extension_length = np.linalg.norm(extension_vector)
    #     if extension_length > 0:
    #         extension_vector = extension_vector / extension_length  # Normalize
    #         roi_center += EXTENSION_FACTOR * extension_length * extension_vector

    #     # Clip to frame size
    #     roi_center[0] = max(0, min(roi_center[0], frame_width - 1))
    #     roi_center[1] = max(0, min(roi_center[1], frame_height - 1))

    #     return tuple(map(int, roi_center))

    def draw_canny(self,center,frame,roi_size):
        x, y = center
        h, w, _ = frame.shape
        x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
        x2, y2 = min(w, x + roi_size), min(h, y + roi_size)
        roi = frame[y1:y2, x1:x2]

        # Convert ROI to grayscale for edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)  # Edge detection
        canny_frame = frame.copy()
        # Expand segmentation outward by detecting contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(canny_frame[y1:y2, x1:x2], [cnt], -1, (0, 255, 0), 2)  # Green boundary

        # Draw rectangle around detected region
        cv2.rectangle(canny_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return canny_frame

    