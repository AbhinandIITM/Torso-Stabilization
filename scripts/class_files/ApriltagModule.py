import dt_apriltags
from dt_apriltags import Detector
import numpy as np
import cv2

class ApriltagModule:
    def __init__(self,tag_size,family,calib_data_path):
        self.tag_size = tag_size
        # TAG_SIZE = 0.05  # meters (5 cm)
        self.at_detector = Detector(
            families= str(family),
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        calib_data = np.load(calib_data_path)
        self.cam_mat = calib_data["camMatrix"]
        self.dist_coef = calib_data["distCoef"]

    def get_tags(self,frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(
            gray_frame, estimate_tag_pose=True,
            camera_params=(self.cam_mat[0, 0], self.cam_mat[1, 1], self.cam_mat[0, 2], self.cam_mat[1, 2]),
            tag_size=self.tag_size
        )
        return tags
    
    def detect_tags(self,frame,tags,display=True):
        results = []
        for tag in tags:
            # Compute depth using translation vector
            tVec = tag.pose_t.flatten()
            apriltag_depth = np.linalg.norm(tVec)

            if display:
                rVec, _ = cv2.Rodrigues(tag.pose_R)

                # Define axis length (in meters)
                axis_length = 0.05

                # Define axis endpoints in 3D (relative to tag origin)
                axis_points = np.float32([
                    [0, 0, 0],  # Origin
                    [axis_length, 0, 0],  # X-axis
                    [0, axis_length, 0],  # Y-axis
                    [0, 0, -axis_length]  # Z-axis (negative since depth is into the screen)
                ]).reshape(-1, 3)

                # Project 3D points to 2D image plane
                img_pts, _ = cv2.projectPoints(axis_points, rVec, tVec, self.cam_mat, self.dist_coef)

                # Convert to integer for drawing
                img_pts = img_pts.astype(int)

                # Draw axes at the tag location
                origin = tuple(img_pts[0].ravel())
                cv2.line(frame, origin, tuple(img_pts[1].ravel()), (0, 0, 255), 2)  # X-axis (red)
                cv2.line(frame, origin, tuple(img_pts[2].ravel()), (0, 255, 0), 2)  # Y-axis (green)
                cv2.line(frame, origin, tuple(img_pts[3].ravel()), (255, 0, 0), 2)  # Z-axis (blue)

                # Label axes
                cv2.putText(frame, "X", tuple(img_pts[1].ravel()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Y", tuple(img_pts[2].ravel()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Z", tuple(img_pts[3].ravel()), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

                # Draw tag bounding box
                for i in range(4):
                    pt1 = tuple(tag.corners[i].astype(int))
                    pt2 = tuple(tag.corners[(i + 1) % 4].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                # Display tag ID and depth
                top_left = tuple(tag.corners[0].astype(int))
                cv2.putText(frame, f"ID:{tag.tag_id} Dist:{round(apriltag_depth * 100, 2)} cm",
                        top_left, cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
            result = {
                        "tag_id": tag.tag_id,
                        "pose_R": tag.pose_R,
                        "pose_t": tag.pose_t,
                        "corners": tag.corners,
                        "center": tag.center,
                        "depth": apriltag_depth
                     }
            results.append(result)

    def get_scaling_factor(self, frame, tags, relative_depth_map):
        scaling_factor = None  # Initialize scaling factor

        for tag in tags:
            # Compute depth using translation vector
            tVec = tag.pose_t.flatten()
            apriltag_depth = np.linalg.norm(tVec)

            # Get rotation vector
            rVec, _ = cv2.Rodrigues(tag.pose_R)

            # Define axis length (in meters)
            axis_length = 0.05  # 5 cm

            # Define axis endpoints in 3D (relative to tag origin)
            axis_points = np.float32([
                [0, 0, 0],  # Origin
                [axis_length, 0, 0],  # X-axis
                [0, axis_length, 0],  # Y-axis
                [0, 0, -axis_length]  # Z-axis (negative depth)
            ]).reshape(-1, 3)

            # Project 3D points to 2D image plane
            img_pts, _ = cv2.projectPoints(axis_points, rVec, tVec, self.cam_mat, self.dist_coef)
            img_pts = img_pts.astype(int)  # Convert to integer for drawing

            # Draw axes on AprilTag
            origin = tuple(img_pts[0].ravel())
            cv2.line(frame, origin, tuple(img_pts[1].ravel()), (0, 0, 255), 2)  # X-axis (Red)
            cv2.line(frame, origin, tuple(img_pts[2].ravel()), (0, 255, 0), 2)  # Y-axis (Green)
            cv2.line(frame, origin, tuple(img_pts[3].ravel()), (255, 0, 0), 2)  # Z-axis (Blue)

            # Label axes
            cv2.putText(frame, "X", tuple(img_pts[1].ravel()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Y", tuple(img_pts[2].ravel()), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Z", tuple(img_pts[3].ravel()), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Draw AprilTag bounding box
            for i in range(4):
                pt1 = tuple(tag.corners[i].astype(int))
                pt2 = tuple(tag.corners[(i + 1) % 4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            # Display tag ID and depth
            top_left = tuple(tag.corners[0].astype(int))
            cv2.putText(frame, f"ID:{tag.tag_id} Dist:{round(apriltag_depth * 100, 2)} cm",
                        top_left, cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)

            # Map tag corners to depth map size
            h, w = frame.shape[:2]
            depth_h, depth_w = relative_depth_map.shape[:2]

            corners = []
            for x, y in tag.corners:
                mapped_x = min(max(int(x * depth_w / w), 0), depth_w - 1)
                mapped_y = min(max(int(y * depth_h / h), 0), depth_h - 1)
                corners.append((mapped_x, mapped_y))

            # Get median relative depth from valid mapped corners
            marker_relative_depth = np.median([relative_depth_map[y, x] for x, y in corners])

            if marker_relative_depth > 0:
                scaling_factor = apriltag_depth / marker_relative_depth

        # print(scaling_factor)
        return apriltag_depth, scaling_factor
