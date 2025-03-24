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
depth0_to_pts_world = (c2w0.unsqueeze(0).unsqueeze(0) @ depth0_to_pts_camera.unsqueeze(-1)).squeeze(-1)


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
mpis = []
depth_maps = []
os.makedirs("mpi_depth", exist_ok=True)
for mask in masks:
    mpis.append((img0 * mask.unsqueeze(-1)).unsqueeze(0))
    depth_maps.append((depth0 * mask.unsqueeze(-1)).unsqueeze(0))
    cv2.imwrite(f"mpi_depth/mpi_img0{count}.png",(img0 * mask.unsqueeze(-1)).cpu().numpy()[..., ::-1])
    pyexr.write(f"mpi_depth/depth_img0{count}.exr", (depth0*mask.unsqueeze(-1)).cpu().numpy())
    count += 1
layered_img = torch.cat(mpis, dim=0)
layered_depth = torch.cat(depth_maps, dim=0)
avg_layered_depth = torch.tensor([0.4, 0.5, 0.7, 2.])


os.makedirs("spiral_mpi_based", exist_ok=True)
trajectory_camera = generate_spiral_trajectory(num_points=100, max_distance=.05, spiral_radius=.2, revolutions=2)
trajectory_camera = torch.cat([trajectory_camera, torch.ones(trajectory_camera.shape[0], 1, device=trajectory_camera.device)], dim=1)
trajectory_world = (c2w0 @ trajectory_camera.T).T
spiral_w2c = create_camera_poses_from_trajectory(trajectory_world, c2w0)


from_img0 = warp_image_homography(layered_img[-1], c2w0, c2w0, K, 0.1)
cv2.imwrite(f"spiral_mpi_based/test.png", ((from_img0*255).cpu().numpy()[..., ::-1]))
for i, w2c in enumerate(spiral_w2c):
    img_list = []
    for rgb, depth in zip(layered_img, avg_layered_depth):
        img_list.append(warp_image_homography(rgb, c2w0, w2c.inverse(), K, depth))
    composition = compose_images(img_list)
    cv2.imwrite(f"spiral_mpi_based/warp_img{i}.png", ((composition*255).cpu().numpy()[..., ::-1]))
