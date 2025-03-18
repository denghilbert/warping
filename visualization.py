from warp import *


# gt
# depth0 = torch.from_numpy(load_exr_opencv("scene0_depth.exr"))
# depth1 = torch.from_numpy(load_exr_opencv("scene1_depth.exr"))
# depth pro
depth0 = torch.from_numpy(dict(np.load("scene0.exr/scene0_perspective_fov90.npz"))['depth']).unsqueeze(-1).repeat(1, 1, 3)
depth1 = torch.from_numpy(dict(np.load("scene1.exr/scene1_perspective_fov90.npz"))['depth']).unsqueeze(-1).repeat(1, 1, 3)

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


# depth0_to_pts_camera, _ = backproject_depth(depth0, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
depth0_to_pts_camera = backproject_depth(depth0, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=False)
depth0_to_pts_world = (c2w0.unsqueeze(0).unsqueeze(0) @ depth0_to_pts_camera.unsqueeze(-1)).squeeze(-1)

# depth1_to_pts_camera, _ = backproject_depth(depth1, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=True)
depth1_to_pts_camera = backproject_depth(depth1, K.inverse(), img0.shape[1], img0.shape[0], if_mitsuba_depth=False)
depth1_to_pts_world = (c2w1.unsqueeze(0).unsqueeze(0) @ depth1_to_pts_camera.unsqueeze(-1)).squeeze(-1)


points = depth0_to_pts_world[..., :3].reshape(-1, 3).numpy()  # Extract XYZ coordinates
colors = img0.reshape(-1, 3).numpy() / 255.0  # Normalize RGB values
# points = depth1_to_pts_world[..., :3].reshape(-1, 3).numpy()  # Extract XYZ coordinates
# colors = img1.reshape(-1, 3).numpy() / 255.0  # Normalize RGB values

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with Color")
