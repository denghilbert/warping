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


depth0_to_pts_camera, depth0_corrected = backproject_depth(depth0, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
depth0_to_pts_world = (c2w0.unsqueeze(0).unsqueeze(0) @ depth0_to_pts_camera.unsqueeze(-1)).squeeze(-1)
depth1_to_pts_camera, _ = backproject_depth(depth1, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
depth1_to_pts_world = (c2w1.unsqueeze(0).unsqueeze(0) @ depth1_to_pts_camera.unsqueeze(-1)).squeeze(-1)

trajectory_camera = generate_spiral_trajectory(num_points=100, max_distance=.05, spiral_radius=.2, revolutions=2)
trajectory_camera = torch.cat([trajectory_camera, torch.ones(trajectory_camera.shape[0], 1, device=trajectory_camera.device)], dim=1)
trajectory_world = (c2w0 @ trajectory_camera.T).T
spiral_w2c = create_camera_poses_from_trajectory(trajectory_world, c2w0)

os.makedirs("spiral", exist_ok=True)
for i, w2c in enumerate(spiral_w2c):
    img_size = (1024, 1024)
    rendered_image = render_image_torch(depth1_to_pts_world, img1, w2c, K, img_size)
    cv2.imwrite(f"spiral/{i}.png", rendered_image.cpu().numpy()[..., ::-1])
