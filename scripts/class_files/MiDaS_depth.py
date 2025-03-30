import torch
import numpy as np
import cv2

class MiDaS_depth():
    def __init__(self):
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = midas.to(self.device)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
        # calib_data = np.load(calib_data_path)
        # self.cam_mat = calib_data["camMatrix"]
        # self.dist_coef = calib_data["distCoef"]
    
    def get_depthmap(self,frame):
        """Estimate relative depth using MiDaS and resize to original frame size."""
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        output = prediction.cpu().numpy()
        output = 1.0 / (output + 1e-6)  # Invert depth map
        return output

    
    # def get_depthmap(self, frame):
    #     """Estimate relative depth using MiDaS and resize to original frame size."""
    #     input_batch = self.transform(frame).to(self.device)
    #     with torch.no_grad():
    #         prediction = self.midas(input_batch)
    #         prediction = torch.nn.functional.interpolate(
    #             prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
    #         ).squeeze()
        
    #     depth_map = (prediction.cpu().numpy())
        
    #     depth_map = 1.0/(1e-6 + depth_map)
    #     # Normalize depth map for visualization
    #     depth_min = depth_map.min()
    #     depth_max = depth_map.max()
    #     depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
    #     depth_map = (depth_map * 255).astype(np.uint8)
    #     # Optional: Apply colormap for better visualization
    #     # print(depth_max)
    #     depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
    #     return depth_map
