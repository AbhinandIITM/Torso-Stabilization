import torch
import numpy as np
import cv2
from PIL import Image

class Zoe_Depth():
    def __init__(self):
        repo = "isl-org/ZoeDepth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384",force_reload=True))  # Triggers fresh download of MiDaS repo
        # Load ZoeD_N
        self.midas = torch.hub.load(repo_or_dir=repo, model="ZoeD_K", pretrained=True,force_reload=True).to(self.device)
        self.midas.eval()

    def get_depthmap(self, frame):
        """
        Estimate relative depth using ZoeDepth.
        Input: BGR frame (OpenCV)
        Output: depth map (numpy array)
        """
        # Convert OpenCV BGR image to RGB PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            # Inference using ZoeDepth
            depth = self.midas.infer_pil(image)  # Returns torch.Tensor (1, H, W)
            depth_np = depth.squeeze().cpu().numpy()

        return depth_np
