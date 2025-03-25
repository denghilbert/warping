from warp import *

depth0 = torch.from_numpy(load_exr_opencv("scene0_depth.exr"))
depth1 = torch.from_numpy(load_exr_opencv("scene1_depth.exr"))
# depth0 = torch.from_numpy(dict(np.load("scene0.exr/scene0_perspective_fov90.npz"))['depth']).unsqueeze(-1).repeat(1, 1, 3)
# depth1 = torch.from_numpy(dict(np.load("scene1.exr/scene1_perspective_fov90.npz"))['depth']).unsqueeze(-1).repeat(1, 1, 3)

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

# projection from image and depth map to 3d
os.makedirs("fast_warping", exist_ok=True)
pts_camera_cam1, depth1_corrected = backproject_depth(depth1, K.inverse(), img1.shape[1], img1.shape[0], if_mitsuba_depth=True)
pts_world_cam1 = (c2w1.unsqueeze(0).unsqueeze(0) @ pts_camera_cam1.unsqueeze(-1)).squeeze(-1)


### how to solve z buffer?
# using depth generate mpi for img0
mpi_depth_camera = torch.tensor([
    [0., 0., 0.5, 1],
    [0., 0., 0.7, 1],
    [0., 0., 2.7, 1],
    [0., 0., 100, 1],
])
mpi_depth_world = (c2w0.unsqueeze(0).unsqueeze(0) @ mpi_depth_camera.unsqueeze(-1)).squeeze(-1).squeeze(0)
planes = mpi_depth_world [:, 0]
planes, _ = torch.sort(planes, descending=True)
x_coords = pts_world_cam1[..., 0]  # Shape: (1024, 1024)
masks = []
for i in range(len(planes) + 1):
    if i == 0:
        mask = x_coords >= planes[i]  # x >= max(planes)
    elif i == len(planes):
        mask = x_coords < planes[i - 1]  # x < min(planes)
    else:
        mask = (x_coords < planes[i - 1]) & (x_coords >= planes[i])  # planes[i] <= x < planes[i-1]
    masks.append(mask)


# warping whole image
# rendered_image = render_image_torch_grid_sample(pts_world_cam1, img1, w2c0, K, (1024, 1024))
rendered_image = render_image_forward_zbuffer(pts_world_cam1, img1, w2c0, K, (1024, 1024))
cv2.imwrite("fast_warping/warp_all.png", rendered_image.cpu().numpy()[..., ::-1])

# warp slices
# points_per_plane = [pts_world_cam1 for mask in masks]
points_per_plane = [torch.where(mask.unsqueeze(-1), pts_world_cam1, mpi_depth_world[-1]) for mask in masks]
rgb_per_plane = [img1 * mask.unsqueeze(-1) for mask in masks]
i = 0
for masked_rgb, masked_pts in zip(rgb_per_plane, points_per_plane):
    cv2.imwrite(f"fast_warping/img1_slice{i}.png", masked_rgb.cpu().numpy()[..., ::-1])
    # rendered_image = render_image_torch_grid_sample(masked_pts, masked_rgb, w2c0, K, (1024, 1024))
    rendered_image = render_image_forward_zbuffer(masked_pts, masked_rgb, w2c0, K, (1024, 1024))
    cv2.imwrite(f"fast_warping/img1_to_img0_slice{i}.png", rendered_image.cpu().numpy()[..., ::-1])
    i += 1
