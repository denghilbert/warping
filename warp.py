import OpenEXR
import Imath
import numpy as np
import cv2
import pyexr
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from PIL import Image
from typing import Tuple


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

def render_image_torch_grid_sample(source_pts, source_rgb, target_w2c, K, img_size=(1024, 1024)):
    """
    Renders an image from homogeneous world points and colors using torch.nn.functional.grid_sample.
    
    Parameters:
        source_pts (torch.Tensor): Homogeneous points of shape (H, W, 4), dtype=torch.float32
        source_rgb (torch.Tensor): Color image of shape (H, W, 3), dtype=torch.uint8
        target_w2c (torch.Tensor): World-to-camera extrinsic matrix (4x4), dtype=torch.float32
        K (torch.Tensor): Intrinsic matrix (3x3), dtype=torch.float32
        img_size (tuple): (height, width) for the rendered image
        
    Returns:
        rendered_img (torch.Tensor): The rendered image with shape (H, W, 3), dtype=torch.uint8
    """
    import torch.nn.functional as F
    
    # Ensure tensors are on the same device
    device = source_pts.device
    source_rgb = source_rgb.to(device)
    target_w2c = target_w2c.to(device)
    K = K.to(device)
    
    H, W = img_size
    H_src, W_src = source_rgb.shape[:2]
    
    # Convert homogeneous points (H, W, 4) to 3D points (H, W, 3)
    pts_xyz = source_pts[..., :3] / torch.clamp(source_pts[..., 3:], min=1e-6)
    
    # Flatten points and convert to homogeneous coordinates
    pts_flat = pts_xyz.reshape(-1, 3)  # (H_src*W_src, 3)
    ones = torch.ones((pts_flat.shape[0], 1), device=device, dtype=torch.float32)
    pts_h_flat = torch.cat([pts_flat, ones], dim=-1)  # (H_src*W_src, 4)
    
    # Transform points from world to camera coordinates
    cam_pts_h_flat = (target_w2c @ pts_h_flat.T).T  # (H_src*W_src, 4)
    cam_pts_flat = cam_pts_h_flat[:, :3]  # (H_src*W_src, 3)
    
    # Project to 2D image plane using intrinsic matrix K
    proj_flat = (K @ cam_pts_flat.T).T  # (H_src*W_src, 3)
    proj_xy_flat = proj_flat[:, :2] / torch.clamp(proj_flat[:, 2:], min=1e-6)  # (H_src*W_src, 2)
    
    # Create source coordinates grid (for sampling source RGB)
    y_src, x_src = torch.meshgrid(torch.arange(H_src, device=device), torch.arange(W_src, device=device))
    source_coords = torch.stack([x_src, y_src], dim=-1).float()  # (H_src, W_src, 2)
    source_coords_flat = source_coords.reshape(-1, 2)  # (H_src*W_src, 2)
    
    # Normalize source coordinates to [-1, 1] for grid_sample
    source_coords_ndc = 2.0 * source_coords_flat / torch.tensor([W_src-1, H_src-1], device=device) - 1.0
    
    # Round projected coordinates to integers and check which ones are valid
    proj_xy_int = torch.round(proj_xy_flat).long()  # (H_src*W_src, 2)
    valid_mask = (proj_xy_int[:, 0] >= 0) & (proj_xy_int[:, 0] < W) & \
                 (proj_xy_int[:, 1] >= 0) & (proj_xy_int[:, 1] < H)
    
    # Filter to keep only valid projections
    valid_proj_xy_int = proj_xy_int[valid_mask]  # (N_valid, 2)
    valid_source_coords = source_coords_ndc[valid_mask]  # (N_valid, 2)
    valid_depths = cam_pts_flat[valid_mask, 2]  # (N_valid,)
    
    # Calculate flat indices for each target pixel
    target_indices = valid_proj_xy_int[:, 1] * W + valid_proj_xy_int[:, 0]  # (N_valid,)
    
    # Sort by depth (front to back) for proper occlusion
    sorted_indices = torch.argsort(valid_depths)
    valid_source_coords = valid_source_coords[sorted_indices]
    target_indices = target_indices[sorted_indices]
    
    # Create empty sampling grid and fill it with coordinates
    sampling_grid = torch.zeros((H*W, 2), device=device)
    
    # The scatter operation places source coordinates at target pixel locations
    # Later points (closer to camera) will overwrite earlier ones
    sampling_grid.scatter_(0, target_indices.unsqueeze(1).repeat(1, 2), valid_source_coords)
    
    # Create valid pixel mask
    valid_pixel_mask = torch.zeros(H*W, dtype=torch.bool, device=device)
    valid_pixel_mask.scatter_(0, target_indices, torch.ones_like(target_indices, dtype=torch.bool))
    
    # Reshape for grid_sample
    sampling_grid = sampling_grid.reshape(1, H, W, 2)
    valid_pixel_mask = valid_pixel_mask.reshape(H, W)
    
    # Prepare source_rgb for grid_sample
    source_rgb_batch = source_rgb.float() / 255.0
    source_rgb_batch = source_rgb_batch.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H_src, W_src)
    
    # Sample colors using grid_sample
    rendered_img = F.grid_sample(
        source_rgb_batch, 
        sampling_grid, 
        mode='nearest',  # Use nearest to match original function
        padding_mode='zeros', 
        align_corners=True
    )
    
    # Convert back to image format
    rendered_img = rendered_img.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    
    # Apply valid pixel mask
    rendered_img = rendered_img * valid_pixel_mask.unsqueeze(-1).float()
    
    # Convert to uint8
    rendered_img = (rendered_img * 255).byte()
    
    return rendered_img

def generate_spiral_trajectory(num_points=100, max_distance=1.0, spiral_radius=0.2, revolutions=2, device='cpu'):
    """
    Generates a spiral trajectory in camera coordinates (x right, y down, z forward).
    
    The trajectory starts at the camera center (0,0,0), spirals forward to a maximum distance,
    and then returns to the camera center.
    
    Args:
        num_points (int): Number of points in the trajectory
        max_distance (float): Maximum forward distance (z) to travel
        spiral_radius (float): Maximum radius of the spiral
        revolutions (float): Number of full revolutions in the spiral
        device (str): PyTorch device ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Trajectory points with shape (num_points, 3)
    """
    # Parameter t goes from 0 to 2 (0->1 for going out, 1->2 for coming back)
    t = torch.linspace(0, 2, num_points, device=device)
    
    # Forward distance (z) follows a sine curve to go out and back
    z = max_distance * torch.sin(torch.pi * t / 2)
    
    # Radius of the spiral varies with t (grows and then shrinks)
    radius = spiral_radius * torch.sin(torch.pi * t)
    
    # Angular position (more revolutions on the way out and back)
    theta = 2 * torch.pi * revolutions * t
    
    # Convert to Cartesian coordinates
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    
    # Stack coordinates to create trajectory
    trajectory = torch.stack([x, y, z], dim=1)
    
    return trajectory

def create_camera_poses_from_trajectory(trajectory, initial_c2w):
    """
    Creates camera poses for each point in a trajectory while maintaining the same look-at direction.
    
    Args:
        trajectory (torch.Tensor): Homogeneous trajectory points with shape (num_points, 4)
        initial_c2w (torch.Tensor): Initial camera-to-world matrix of shape (4, 4)
        
    Returns:
        torch.Tensor: Camera-to-world matrices for each trajectory point, shape (num_points, 4, 4)
    """
    num_points = trajectory.shape[0]
    device = trajectory.device
    
    # Extract rotation part (first 3x3) from initial camera matrix
    R = initial_c2w[:3, :3]
    
    # Create empty tensor to store all camera poses
    w2c_matrices = torch.zeros((num_points, 4, 4), device=device)
    
    # For each point in the trajectory
    for i in range(num_points):
        # Create new c2w matrix
        c2w = torch.eye(4, device=device)
        
        # Copy rotation (keep the same orientation/look-at direction)
        c2w[:3, :3] = R
        
        # Set the new camera center (translation)
        c2w[:3, 3] = trajectory[i, :3]
        
        # Store in the result tensor
        w2c_matrices[i] = c2w.inverse()
    
    return w2c_matrices

def warp_image_homography(src_img, src_c2w, dst_c2w, K, depth):
    """
    Warp an image from source camera to destination camera using homography.
    
    Args:
        src_img: Tensor of shape [H, W, 3] representing the source image.
        src_c2w: Tensor of shape [4, 4] representing camera-to-world transformation for source camera.
        dst_c2w: Tensor of shape [4, 4] representing camera-to-world transformation for destination camera.
        K: Tensor of shape [3, 3] representing camera intrinsic matrix (assumed same for both cameras).
        depth: Float representing the distance from source camera to the image plane.
        
    Returns:
        warped_img: Tensor of shape [H, W, 3] representing the warped image.
    """
    height, width, _ = src_img.shape
    device = src_img.device
    
    # Convert src_img to float if it's not already
    if src_img.dtype != torch.float32:
        src_img = src_img.float()
        
        # If it was originally uint8 (0-255), normalize to 0-1 range
        if src_img.max() > 1.0:
            src_img = src_img / 255.0
    
    # Convert camera-to-world to world-to-camera
    src_w2c = torch.inverse(src_c2w)  # Source world-to-camera
    dst_w2c = torch.inverse(dst_c2w)  # Destination world-to-camera
    
    # Extract rotation and translation for both cameras
    R_src = src_w2c[:3, :3]  # Rotation for source camera
    t_src = src_w2c[:3, 3]   # Translation for source camera
    R_dst = dst_w2c[:3, :3]  # Rotation for destination camera
    t_dst = dst_w2c[:3, 3]   # Translation for destination camera
    
    # Calculate relative transformation from source to destination
    R_rel = R_dst @ torch.inverse(R_src)  # Relative rotation
    t_rel = t_dst - R_rel @ t_src         # Relative translation
    
    # Plane normal in source camera coordinates (along z-axis)
    n = torch.tensor([0., 0., 1.], device=device)
    
    # Compute the homography matrix
    # H = K_dst * R_rel * (I - t_rel*n^T/d) * K_src^-1
    I = torch.eye(3, device=device)
    H = K @ R_rel @ (I - torch.outer(t_rel, n) / depth) @ torch.inverse(K)

    
    # Create pixel coordinates for the source image
    xs = torch.arange(width, device=device)
    ys = torch.arange(height, device=device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
    
    # Create homogeneous coordinates [x, y, 1] for each pixel
    ones = torch.ones_like(x_grid)
    homo_coords = torch.stack([x_grid, y_grid, ones], dim=0).float()  # [3, H, W]
    
    # Apply the homography to transform pixel coordinates
    # from source image to destination image
    warped_coords = H @ homo_coords.reshape(3, -1)  # [3, H*W]
    warped_coords = warped_coords.reshape(3, height, width)
    
    # Normalize the homogeneous coordinates
    z = warped_coords[2:3, :, :] + 1e-8
    warped_coords = warped_coords / z
    
    # Convert to [-1, 1] range for grid_sample
    grid_x = 2.0 * warped_coords[0] / (width - 1) - 1.0
    grid_y = 2.0 * warped_coords[1] / (height - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    
    # Check valid coordinates
    valid_mask = ((grid_x >= -1.0) & (grid_x <= 1.0) & 
                  (grid_y >= -1.0) & (grid_y <= 1.0) & 
                  (z.squeeze(0) > 0))
    
    # Warp the image using grid_sample
    img_tensor = src_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    warped_img = F.grid_sample(img_tensor, grid.unsqueeze(0), 
                              mode='bilinear', 
                              padding_mode='zeros', 
                              align_corners=True)
    warped_img = warped_img[0].permute(1, 2, 0)  # [H, W, 3]
    
    # Apply valid mask
    warped_img = warped_img * valid_mask.unsqueeze(-1).float()
    
    # Convert back to original data type if needed
    if src_img.dtype == torch.uint8:
        warped_img = (warped_img * 255.0).to(torch.uint8)
    
    return warped_img

def compose_images(img_list):
    """
    Compose multiple images into a single image, ensuring img_list[0] is in front,
    followed by img_list[1], img_list[2], etc.
    
    Args:
        img_list: List of tensor images, each with shape [H, W, 3]
        
    Returns:
        composite: Tensor with shape [H, W, 3] representing the composited image
    """
    if not img_list:
        raise ValueError("Image list cannot be empty")
    
    height, width, channels = img_list[0].shape
    device = img_list[0].device
    dtype = img_list[0].dtype
    
    # Initialize composite with zeros
    composite = torch.zeros(height, width, channels, device=device, dtype=torch.float32)
    
    # Create a blank alpha mask to track occupied pixels
    alpha_mask = torch.zeros(height, width, 1, device=device, dtype=torch.float32)
    
    # Process images front to back (img_list[0] first, ensures it's on top)
    for i in range(len(img_list)):
        img = img_list[i]
        
        # Convert to float for processing if needed
        if img.dtype != torch.float32:
            img_float = img.float()
            if img.dtype == torch.uint8:
                img_float = img_float / 255.0
        else:
            img_float = img
        
        # Calculate the alpha (where image has content)
        img_alpha = (img_float.sum(dim=2, keepdim=True) > 0).float()
        
        # For the first image (index 0), directly add it to the composite
        if i == 0:
            composite = img_float * img_alpha
            alpha_mask = img_alpha.clone()
        else:
            # For subsequent images, only add where the alpha mask is still empty
            available_alpha = 1.0 - alpha_mask
            
            # Update composite image with new pixels
            composite = composite + img_float * img_alpha * available_alpha
            
            # Update alpha mask
            alpha_mask = alpha_mask + (img_alpha * available_alpha)
    
    # Convert back to original dtype if needed
    if dtype != torch.float32:
        if dtype == torch.uint8:
            composite = (composite * 255.0).to(torch.uint8)
        else:
            composite = composite.to(dtype)
    
    return composite
