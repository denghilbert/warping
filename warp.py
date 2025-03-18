import OpenEXR
import Imath
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


def to_homogeneous(points):
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device, requires_grad=True)
    points_hom = torch.cat([points, ones], dim=1)  # Concatenate along last dimension
    return points_hom

def load_exr_opencv(file_path):
    """Load a 3-channel EXR file using OpenCV."""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Load EXR
    if img is None:
        raise ValueError(f"Failed to load EXR file: {file_path}")
    
    # Ensure it's a 3-channel image
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)  # Convert grayscale to 3-channel format
    elif img.shape[-1] == 4:  # If 4-channel (RGBA), remove alpha
        img = img[:, :, :3]

    return img.astype(np.float32)

def load_jpg(file_path):
    """Load a JPG file as a NumPy array."""
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img

def get_intrinsic_matrix(W, H, camera_angle_x):
    camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32)  # Ensure it's a tensor
    fx = W / (2 * torch.tan(camera_angle_x / 2))
    fy = fx  # Assuming square pixels

    K = torch.tensor([
        [fx, 0, W / 2],
        [0, fy, H / 2],
        [0,  0,  1]
    ], dtype=torch.float32)

    return K

def project_3d_to_2d(points, K, w2c, eps=1e-7):
    """
    Projects 3D world points into pixel coordinates using intrinsic and extrinsic matrices.

    Args:
        points (torch.Tensor): (N, 4) tensor of homogeneous 3D points in world coordinates.
        K (torch.Tensor): (3, 3) intrinsic matrix.
        w2c (torch.Tensor): (3, 4) world-to-camera extrinsic matrix.
        eps (float): Small constant to avoid division by zero.

    Returns:
        pix_coords (torch.Tensor): (N, 2) tensor of 2D pixel coordinates.
    """
    cam_pts = K @ (w2c @ points.T)  # Shape: (3, N)
    pix_coords = cam_pts[:2] / (cam_pts[2].unsqueeze(0) + eps)  # Shape: (2, N)

    return pix_coords.T  # Return as (N, 2)

def backproject_depth(depth, inv_K, height, width, if_mitsuba_depth=False):
    """
    Transforms a depth image into a 3D point cloud in camera coordinates
    
    Args:
        depth (torch.Tensor): (H, W, 3) depth image.
        inv_K (torch.Tensor): (3, 3) inverse intrinsic matrix.
        height (int): Image height.
        width (int): Image width.
        if_mitsuba_depth: mitsuba depth is the distance from camera center to points
    
    Returns:
        cam_points (torch.Tensor): (H, W, 4) 3D points in homogeneous coordinates.
    """
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32, device=depth.device),
                          torch.arange(width, dtype=torch.float32, device=depth.device),
                          indexing='xy')
    
    pix_coords = torch.stack([y, x, torch.ones_like(x)], dim=0)  # (3, H, W)
    cam_points = torch.einsum('ij,jhw->ihw', inv_K, pix_coords).permute(1, 2, 0)  # (H, W, 3)

    if if_mitsuba_depth:
        mitsuba_denominator = torch.sqrt((cam_points ** 2).sum(dim=-1)).unsqueeze(-1)
        depth = depth / mitsuba_denominator

    cam_points = depth * cam_points  # (H, W, 3)
    cam_points = torch.cat([cam_points, torch.ones_like(cam_points[..., :1])], dim=-1)
    
    return cam_points

depth0 = torch.from_numpy(load_exr_opencv("scene0_depth.exr"))
depth1 = torch.from_numpy(load_exr_opencv("scene1_depth.exr"))
img0 = torch.from_numpy(load_jpg("scene0_perspective_fov90.jpg"))
img1 = torch.from_numpy(load_jpg("scene1_perspective_fov90.jpg"))
depths = torch.cat((depth0.unsqueeze(0), depth1.unsqueeze(0)), dim=0)

json_path = 'transforms_test_mpi.json'
if not os.path.exists(json_path):
    print(f"Error: JSON file not found: {json_path}")
with open(json_path, 'r') as f:
    data = json.load(f)

camera_angle_x = data['camera_angle_x']
K = get_intrinsic_matrix(img0.shape[1], img0.shape[0], camera_angle_x)
c2w0 = torch.tensor(data['frames'][0]['transform_matrix']) # original x left y up z forward
c2w0[:, :2] *= -1 # x right y down z forward
c2w1 = torch.tensor(data['frames'][1]['transform_matrix'])
c2w1[:, :2] *= -1
w2c0 = c2w0.inverse()[:3, :]
w2c1 = c2w1.inverse()[:3, :]


test_3d_pt = to_homogeneous(torch.tensor([[-2.465, 1.385, -1.5635], 
                                          [-2.3968,  3.2240, 3.6474],
                                          [-2.3968,  3.2240, -1.8374],
                                          [-1.6331, 0.90826, 0.33563]]))

test_pts = project_3d_to_2d(test_3d_pt, K, w2c1)
print(test_pts)

depth2pts_camera = backproject_depth(depth1, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
depth2pts_world = (c2w1.unsqueeze(0).unsqueeze(0) @ depth2pts_camera.unsqueeze(-1)).squeeze(-1)
pix_pts = project_3d_to_2d(depth2pts_world.reshape(-1, 4), K, w2c1).reshape(img0.shape[1], img0.shape[0], 2)

print("*"*10)
print(depth2pts_world[357][878])
print(depth2pts_world[0][0])
print(depth2pts_world[0][1023])
print(depth2pts_world[457][615])
import pdb;pdb.set_trace()
