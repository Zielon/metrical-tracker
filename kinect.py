import torch
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.utils import cameras_from_opencv_projection


class Intrinsics:
    def __init__(self, K):
        pp_x, pp_y, fl_x, fl_y, w, h = K
        self.pp_x = pp_x
        self.pp_y = pp_y
        self.fl_y = fl_y
        self.fl_x = fl_x
        self.image_size = torch.tensor([[h, w]]).cuda()

    def get_camera(self):
        R, t = look_at_view_transform(device='cuda:0')
        K = torch.tensor([[[self.fl_x, 0, self.pp_x], [0, self.fl_y, self.pp_y], [0, 0, 1]]]).cuda()
        camera = cameras_from_opencv_projection(R, t, K, self.image_size)
        camera.R = R
        camera.T = t
        return camera


class Kinect:
    def __init__(self, color=None, depth=None):
        self.color_intrinsics = color
        self.is_in_crop_space = False
        # self.depth_intrinsics = depth

    def adjust_camera_to_crop(self, bbox):
        cx = self.color_intrinsics.pp_x
        cy = self.color_intrinsics.pp_y
        x = bbox['xb_min']
        y = bbox['yb_min']
        cx = cx - x
        cy = cy - y
        self.color_intrinsics.pp_x = cx
        self.color_intrinsics.pp_y = cy
        self.color_intrinsics.image_size = torch.tensor([[bbox['yb_max'] - bbox['yb_min'], bbox['xb_max'] - bbox['xb_min']]]).cuda()
        self.is_in_crop_space = True
