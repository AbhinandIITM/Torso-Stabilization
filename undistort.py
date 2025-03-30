import cv2
import numpy as np

# === Camera Intrinsics ===
cam_mat = np.array([[876.58388117, 0., 988.52249219],
                    [0., 875.69167147, 527.50461067],
                    [0., 0., 1.]])

dist_coef = np.array([[-0.0212738, 0.05926712, -0.00043674, 0.00102205, -0.06370663]])

# === Start Video Capture ===
cap = cv2.VideoCapture(2)  # Change index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]  # Image dimensions

    # Generate grid of pixel coordinates
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Stack into (N, 1, 2) shape
    pixel_coords = np.stack((x_grid, y_grid), axis=-1).reshape(-1, 1, 2)

    # Compute normalized coordinates
    normalized_pts = cv2.undistortPoints(pixel_coords, cam_mat, dist_coef)

    # Extract x' and y' (normalized values between -1 and 1)
    x_norm = normalized_pts[:, 0, 0].reshape(h, w)
    y_norm = -normalized_pts[:, 0, 1].reshape(h, w)

    # Map to color range (0 to 255)
    red_channel = ((x_norm + 1) / 2 * 255).astype(np.uint8)  # Map x' to (0,255)
    green_channel = ((y_norm + 1) / 2 * 255).astype(np.uint8)  # Map y' to (0,255)
    blue_channel = np.zeros_like(red_channel)  # No blue component

    # Merge channels into an image
    color_map = cv2.merge((blue_channel, green_channel, red_channel))

    # Display the colored normalized coordinate map
    cv2.imshow("Normalized (X, Y) Color Map", color_map)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite('colormap.png',color_map)
# Release resources
cap.release()
cv2.destroyAllWindows()
