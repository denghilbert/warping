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

# generate mpi with different depth
mpi_depth_camera = torch.tensor([
    [0., 0., 0.5, 1],
    [0., 0., 0.7, 1],
    [0., 0., 2., 1],
])
mpi_depth_world = (c2w0.unsqueeze(0).unsqueeze(0) @ mpi_depth_camera.unsqueeze(-1)).squeeze(-1).squeeze(0)
planes = mpi_depth_world [:, 0]
planes, _ = torch.sort(planes, descending=True)

depth0_to_pts_camera, depth0_corrected = backproject_depth(depth0, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
# depth0_to_pts_camera = backproject_depth(depth0, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=False)
depth0_to_pts_world = (c2w0.unsqueeze(0).unsqueeze(0) @ depth0_to_pts_camera.unsqueeze(-1)).squeeze(-1)
img_size = (1024, 1024)
rendered_image = render_image_torch(depth0_to_pts_world, img0, w2c1, K, img_size)
cv2.imwrite("img0_to_img1.png", rendered_image.cpu().numpy()[..., ::-1])

# using depth generate mpi for img0
x_coords = depth0_to_pts_world[..., 0]  # Shape: (1024, 1024)
masks = []
for i in range(len(planes) + 1):
    if i == 0:
        mask = x_coords >= planes[i]  # x >= max(planes)
    elif i == len(planes):
        mask = x_coords < planes[i - 1]  # x < min(planes)
    else:
        mask = (x_coords < planes[i - 1]) & (x_coords >= planes[i])  # planes[i] <= x < planes[i-1]
    masks.append(mask)
count = 0
for mask in masks:
    cv2.imwrite(f"mpi_img0{count}.png", (img0*mask.unsqueeze(-1)).cpu().numpy()[..., ::-1])
    count += 1


depth1_to_pts_camera, _ = backproject_depth(depth1, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
# depth1_to_pts_camera = backproject_depth(depth1, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=False)
depth1_to_pts_world = (c2w1.unsqueeze(0).unsqueeze(0) @ depth1_to_pts_camera.unsqueeze(-1)).squeeze(-1)

x_coords = depth1_to_pts_world[..., 0]  # Shape: (1024, 1024)
masks = []
for i in range(len(planes) + 1):
    if i == 0:
        mask = x_coords >= planes[i]  # x >= max(planes)
    elif i == len(planes):
        mask = x_coords < planes[i - 1]  # x < min(planes)
    else:
        mask = (x_coords < planes[i - 1]) & (x_coords >= planes[i])  # planes[i] <= x < planes[i-1]
    masks.append(mask)
points_per_plane = [depth1_to_pts_world[mask] for mask in masks]
rgb_per_plane = [img1[mask] for mask in masks]

img_size = (1024, 1024)
count = 0
for pts, rgb in zip(points_per_plane, rgb_per_plane):
    rendered_image = render_image_torch(pts, rgb, w2c0, K, img_size)
    cv2.imwrite(f"img1_to_img0_plane{count}.png", rendered_image.cpu().numpy()[..., ::-1])
    count += 1

