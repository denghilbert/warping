import OpenEXR
import Imath
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from PIL import Image


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

    w2c coordinates: x right y down z forward

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
    coordinates: x right y down z forward
    
    Args:
        depth (torch.Tensor): (H, W, 3) depth image.
        inv_K (torch.Tensor): (3, 3) inverse intrinsic matrix.
        height (int): Image height.
        width (int): Image width.
        if_mitsuba_depth: mitsuba depth is the distance from camera center to points
    
    Returns:
        cam_points (torch.Tensor): (H, W, 4) 3D points in homogeneous coordinates.
        if_mitsuba_depth, depth_correct: (H, W, 3) the corrected depth map
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
    
    if if_mitsuba_depth:
        return cam_points, depth
    else:
        return cam_points

def warp_image(img, pts_mapping, mode='bilinear', align_corners=False):
    """
    Warp an image using PyTorch's grid_sample function.
    Out-of-bounds coordinates (< 0 or >= H,W) will be set to black.
    
    Args:
        img (torch.Tensor): Input image tensor with shape [H, W, C]
        pts_mapping (torch.Tensor): Coordinate mapping tensor with shape [H, W, 2]
        mode (str): Interpolation mode, one of 'bilinear', 'nearest', or 'bicubic'
        align_corners (bool): Whether to align corners when normalizing coordinates
        
    Returns:
        torch.Tensor: Warped image with the same shape as input image
    """
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Convert image to float if it's not already
    original_dtype = img.dtype
    if img.dtype != torch.float32:
        img_float = img.float()
        # If the image is in byte format (0-255), normalize to 0-1
        if img.dtype == torch.uint8:
            img_float = img_float / 255.0
    else:
        img_float = img
    
    # Reshape img to match the expected input format for grid_sample [N, C, H, W]
    img_tensor = img_float.permute(2, 0, 1).unsqueeze(0)  # Shape becomes [1, C, H, W]
    
    # Normalize pts_mapping to [-1, 1] range as required by grid_sample
    pts_normalized = pts_mapping.clone()
    pts_normalized[..., 0] = 2 * pts_normalized[..., 0] / (w - 1) - 1  # x coordinates
    pts_normalized[..., 1] = 2 * pts_normalized[..., 1] / (h - 1) - 1  # y coordinates
    
    # Add batch dimension to the grid
    grid = pts_normalized.unsqueeze(0)  # Shape [1, H, W, 2]
    
    # Use grid_sample to warp the image
    warped_img = F.grid_sample(
        img_tensor, grid, 
        mode=mode, 
        padding_mode='zeros',  # Set out-of-bounds pixels to black
        align_corners=align_corners
    )
    
    # Convert back to the original format [H, W, C]
    warped_img = warped_img.squeeze(0).permute(1, 2, 0)
    
    # Convert back to original dtype if needed
    if original_dtype == torch.uint8:
        warped_img = (warped_img * 255.0).clamp(0, 255).to(torch.uint8)
    
    return warped_img

def render_image_torch(source_pts, source_rgb, target_w2c, K, img_size=(1024, 1024)):
    """
    Renders an image from homogeneous world points and colors using the provided extrinsics and intrinsics.
    
    Parameters:
        source_pts (torch.Tensor): Homogeneous points of shape (H, W, 4), dtype=torch.float32
        source_rgb (torch.Tensor): Color image of shape (H, W, 3), dtype=torch.uint8
        target_w2c (torch.Tensor): World-to-camera extrinsic matrix (4x4), dtype=torch.float32
        K (torch.Tensor): Intrinsic matrix (3x3), dtype=torch.float32
        img_size (tuple): (height, width) for the rendered image
        
    Returns:
        rendered_img (torch.Tensor): The rendered image with shape (H, W, 3), dtype=torch.uint8
    """
    
    # Ensure tensors are on the same device
    device = source_pts.device
    source_rgb = source_rgb.to(device)
    target_w2c = target_w2c.to(device)
    K = K.to(device)

    # Convert homogeneous points (H, W, 4) to 3D points (H, W, 3)
    pts_xyz = source_pts[..., :3] / torch.clamp(source_pts[..., 3:], min=1e-6)

    # Flatten (N, 3) and (N, 3)
    H, W = img_size
    points = pts_xyz.reshape(-1, 3)  # (H*W, 3)
    colors = source_rgb.reshape(-1, 3).float() / 255.0  # Normalize to [0,1]

    # Convert to homogeneous coordinates (N, 4)
    ones = torch.ones((points.shape[0], 1), device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=-1)  # (N, 4)

    # Transform points from world to camera coordinates
    cam_points_h = (target_w2c @ points_h.T).T  # (N, 4)
    cam_points = cam_points_h[:, :3]  # (N, 3)

    # Project to 2D image plane using intrinsic matrix K
    proj = (K @ cam_points.T).T  # (N, 3)
    proj_xy = proj[:, :2] / torch.clamp(proj[:, 2:], min=1e-6)  # Normalize by depth
    proj_xy_int = proj_xy.round().long()  # Convert to integer pixel coordinates

    # Initialize rendered image and depth buffer
    rendered_img = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
    depth_buffer = torch.full((H, W), float('inf'), dtype=torch.float32, device=device)

    # Rasterization with depth test
    for i in range(proj_xy_int.shape[0]):
        u, v = proj_xy_int[i]
        if 0 <= u < W and 0 <= v < H:
            depth = cam_points[i, 2]  # Z-depth
            if depth < depth_buffer[v, u]:  # Closer point wins
                rendered_img[v, u] = (colors[i] * 255).byte()
                depth_buffer[v, u] = depth

    return rendered_img
