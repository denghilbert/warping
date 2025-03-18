import warp
from warp import get_intrinsic_matrix, to_homogeneous, project_3d_to_2d, backproject_depth


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
