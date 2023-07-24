#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import os
import numpy as np
import open3d as o3d
import argparse
from reconstruct.utils import color_table, set_view, get_configs, get_decoder
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer, MeshExtractor
import numpy as np

import plyfile

angle = 90
angle_rad = np.deg2rad(angle)
rot_x_world = np.array([
    [1, 0, 0, 0],
    [0, np.cos(-angle_rad), -np.sin(-angle_rad), 0],
    [0, np.sin(-angle_rad),  np.cos(-angle_rad), 0],
    [0, 0, 0, 1]
])

rot_y_world = np.array([
    [np.cos(angle_rad), 0, -np.sin(angle_rad), 0],
    [0,             1,              0, 0],
    [np.sin(angle_rad), 0,  np.cos(angle_rad), 0],
    [0, 0, 0, 1]
])

rot_z_world = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad),0 ,0],
    [np.sin(angle_rad),  np.cos(angle_rad),0 ,0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

scale = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad),0 ,0],
    [np.sin(angle_rad),  np.cos(angle_rad),0 ,0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

scale_pam = 1./2.4
scale = scale_pam * np.array(np.eye(3))
    
rott = rot_x_world @ rot_z_world
scale_mtx = scale @ rott[:3, :3]

rott_temp = np.hstack((scale_mtx, rott[0:3, 3][np.newaxis].T)) 
rott = np.vstack((rott_temp, rott[3]))    
print("rott", rott)                                             

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    return parser


def write_mesh_to_ply(v, f, ply_filename_out):
    # try writing to the ply file

    num_verts = v.shape[0]
    num_faces = f.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(v[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((f[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


# 2D and 3D detection and data association
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    configs = get_configs(args.config)
    decoder = get_decoder(configs)
    optimizer = Optimizer(decoder, configs)

    # start reconstruction
    objects_recon = []
    start = get_time()
        
    canonical_points = np.load('canonical_points.npy', allow_pickle='TRUE').item()
    cars = len(canonical_points)
    all_points = np.array([[0, 0, 0]])
    for i in range(cars):
        try:
            all_points = np.concatenate((all_points, canonical_points[i]))
        except KeyError:
            pass
    all_points = all_points[1:]
    
    all_points = np.hstack((all_points, np.ones((all_points.shape[0], 1))))
    rot_points = (rott @ all_points.T).T
    all_points = rot_points[:, :3]
    
    obj = optimizer.reconstruct_object(np.eye(4, dtype="float32"), all_points)
    objects_recon = [obj]


    end = get_time()
    # print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_ctr = vis.get_view_control()
    
    # Add SOURCE LiDAR point cloud
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(all_points)
    green_color = np.full((all_points.shape[0], 3), color_table[1]) # Green COLOR
    scene_pcd.colors = o3d.utility.Vector3dVector(green_color)
    vis.add_geometry(scene_pcd)
    
    # Create mesh extractor
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for i, obj in enumerate(objects_recon):
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)

        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        # red_pcd = np.save("red_pcd.npy", mesh.vertices)

        # Add OUTPUT LiDAR point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        red_color = np.full((mesh.vertices.shape[0], 3), color_table[0]) # Red COLOR
        obj_pcd.colors = o3d.utility.Vector3dVector(red_color)
        vis.add_geometry(obj_pcd)
        

        mesh_o3d.compute_vertex_normals()
        
        # Transform mesh from object to world coordinate
        # mesh_o3d.transform(obj.t_cam_obj)
        # obj_pcd.transform(obj.t_cam_obj)
        
        vis.add_geometry(mesh_o3d)
        print("AFTER obj.t_cam_obj", obj.t_cam_obj)


    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    coordinate_frame.transform(rott)
    # scene_pcd.transform(rott)
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()



# python reconstruct_frame.py --config configs/config_kitti.json