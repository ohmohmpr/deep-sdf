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

import time
import numpy as np
import open3d as o3d
import argparse
from reconstruct.utils import color_table, set_view, get_configs, get_decoder
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer, MeshExtractor
import numpy as np
import plyfile


RED = color_table[0]
GREEN = color_table[1]
BLUE = color_table[2]

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

def convert_points_to_homo(point3D):
    points = np.hstack((point3D, np.ones((point3D.shape[0], 1))))
    rot_points = (rott @ points.T).T
    points = rot_points[:, :3]
    return points
    
def add_coordinate_frame(vis):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    coordinate_frame.transform(rott)
    vis.add_geometry(coordinate_frame)

def set_view(vis, dist=100., theta=np.pi/2.):
    """
    :param vis: o3d visualizer
    :param dist: eye-to-world distance, assume eye is looking at world origin
    :param theta: tilt-angle around x-axis of world coordinate
    """
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    # world to eye
    X = np.array([
                [np.cos(theta), -np.sin(theta), 0., 0.],
                [np.sin(theta), np.cos(theta), 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
    ])
    
    Y_angle = np.radians(90)
    Y = np.array([
                [np.cos(Y_angle), 0., -np.sin(Y_angle), 0.],
                [0., 1., 0., 0.],
                [np.sin(Y_angle), 0., np.cos(Y_angle), dist],
                [0., 0., 0., 1.]
    ])
    
    
    # Z_angle = np.radians(30)
    # Z = np.array([
    #             [1., 0., 0., 0.],
    #             [0., np.cos(Z_angle), -np.sin(Z_angle), 0.],
    #             [0., np.sin(Z_angle), np.cos(Z_angle), 0.],
    #             [0., 0., 0., 1.]
    # ])
    
    T = X @ Y # Not cumulative
    cam.extrinsic = T
    vis_ctr.convert_from_pinhole_camera_parameters(cam)
    
# 2D and 3D detection and data association
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    configs = get_configs(args.config)
    decoder = get_decoder(configs)
    optimizer = Optimizer(decoder, configs)

    # start reconstruction
    objects_recon = []
    source_points = []
    
    objects_recon_latent = []

    frames = []
    
    start = get_time()
        
    canonical_points = np.load('canonical_points.npy', allow_pickle='TRUE').item()
    cars = len(canonical_points)

    thes_hold_num_pcd = 200
    

    latent_vector = np.zeros(64)
    for i in range(cars):
        try:
            if canonical_points[i].shape[0] > thes_hold_num_pcd:
                points = convert_points_to_homo(canonical_points[i])
                obj = optimizer.reconstruct_object(np.eye(4, dtype="float32"), points)
                frames += [i]
                objects_recon += [obj]
                source_points += [points]
                
                obj_latent = optimizer.reconstruct_object(np.eye(4, dtype="float32"), points, latent_vector)
                objects_recon_latent += [obj_latent]
                latent_vector = obj_latent.code
        except KeyError:
            pass
        

    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis_latent = o3d.visualization.Visualizer()
    
    vis.create_window(window_name='Zero initialize latent code', width=960, height=540, left=50, top=50)
    vis_latent.create_window(window_name='use the last one as guess value', width=960, height=540, left=750, top=50)
    
    vis_ctr = vis.get_view_control()
    vis_latent_ctr = vis_latent.get_view_control()
    
    # Add empty object
    mesh_o3d = o3d.geometry.TriangleMesh()
    source_pcd = o3d.geometry.PointCloud()
    output_pcd = o3d.geometry.PointCloud()
    
    mesh_o3d_latent = o3d.geometry.TriangleMesh()
    source_pcd_latent = o3d.geometry.PointCloud()
    output_pcd_latent = o3d.geometry.PointCloud()
    
    vis.add_geometry(mesh_o3d, reset_bounding_box=False)
    vis.add_geometry(source_pcd, reset_bounding_box=False)
    vis.add_geometry(output_pcd, reset_bounding_box=False)
    
    vis_latent.add_geometry(mesh_o3d_latent, reset_bounding_box=False)
    vis_latent.add_geometry(source_pcd_latent, reset_bounding_box=False)
    vis_latent.add_geometry(output_pcd_latent, reset_bounding_box=False)
    
    # Add coordinate frame
    add_coordinate_frame(vis)
    add_coordinate_frame(vis_latent)
    set_view(vis, dist=2., theta=np.radians(180))
    set_view(vis_latent, dist=2., theta=np.radians(180))
    
    # Create mesh extractor
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for frame, source_point, obj, obj_latent in zip(frames, source_points, objects_recon, objects_recon_latent):
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_latent = mesh_extractor.extract_mesh_from_code(obj_latent.code)

        # Update Mesh
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()
        vis.update_geometry(mesh_o3d)
        
        # Update mesh_latent
        mesh_o3d_latent.vertices = o3d.utility.Vector3dVector(mesh_latent.vertices)
        mesh_o3d_latent.triangles = o3d.utility.Vector3iVector(mesh_latent.faces)
        mesh_o3d_latent.compute_vertex_normals()
        vis_latent.update_geometry(mesh_o3d_latent)
        
        # Update Source Point Cloud
        source_pcd.points = o3d.utility.Vector3dVector(source_point)
        source_pcd.paint_uniform_color(GREEN)
        vis.update_geometry(source_pcd)
        
        # Update Source Point Cloud
        source_pcd_latent.points = o3d.utility.Vector3dVector(source_point)
        source_pcd_latent.paint_uniform_color(GREEN)
        vis_latent.update_geometry(source_pcd_latent)
        
        # Update Output LiDAR Point Cloud - FIX BUG to pass by reference (convert to function)
        output_pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        output_pcd.paint_uniform_color(RED)
        vis.update_geometry(output_pcd)
        
        # Update Output LiDAR Point Cloud - FIX BUG to pass by reference (convert to function)
        output_pcd_latent.points = o3d.utility.Vector3dVector(mesh_latent.vertices)
        output_pcd_latent.paint_uniform_color(BLUE)
        vis_latent.update_geometry(output_pcd_latent)
        
        # Print
        print("\n")
        print("FRAME", frame)
        print("Number of point cloud", source_point.shape[0])
        print("Optimized transformation", obj.t_cam_obj)
        print("Different in transformation", np.abs(obj.t_cam_obj - obj_latent.t_cam_obj))
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        
        # Render
        vis_latent.poll_events()
        vis_latent.update_renderer()
        # time.sleep(0.5)
        
    vis.destroy_window()
    vis_latent.destroy_window()

#  python reconstruct_each_frame.py --config configs/config_kitti.json