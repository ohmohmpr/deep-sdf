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
from reconstruct.kitti_sequence import KITIISequence
from reconstruct.optimizer import Optimizer, MeshExtractor
import numpy as np

import plyfile

rot_x_world = np.array([
    [1, 0, 0, 0],
    [0, np.cos(1.57), -np.sin(1.57), 0],
    [0, np.sin(1.57),  np.cos(1.57), 0],
    [0, 0, 0, 1]
])

rot_y_world = np.array([
    [np.cos(1.57), 0, -np.sin(1.57), 0],
    [0,             1,              0, 0],
    [np.sin(1.57), 0,  np.cos(1.57), 0],
    [0, 0, 0, 1]
])

rot_z_world = np.array([
    [np.cos(-1.57), -np.sin(-1.57),0 ,0],
    [np.sin(-1.57),  np.cos(-1.57),0 ,0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


rott = rot_x_world @ rot_y_world

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    # parser.add_argument('-d', '--sequence_dir', type=str, required=True, help='path to kitti sequence')
    # parser.add_argument('-i', '--frame_id', type=int, required=True, help='frame id')
    return parser

# python reconstruct_frame.py --config configs/config_kitti.json

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
    # kitti_seq = KITIISequence(args.sequence_dir, configs)
    optimizer = Optimizer(decoder, configs)
    # detections = kitti_seq.get_frame_by_id(args.frame_id)

    # start reconstruction
    objects_recon = []
    start = get_time()
    # for det in detections:
        # No observed rays, possibly not in fov
        # if det.rays is None:
        #     continue
        # print("%d depth samples on the car, %d rays in total" % (det.num_surface_points, det.rays.shape[0]))
        # obj = optimizer.reconstruct_object(det.T_cam_obj, det.surface_points, det.rays, det.depth)
        # # in case reconstruction fails
        # if obj.code is None:
        #     continue
        # objects_recon += [obj]
    canonical_points = np.load('canonical_points.npy', allow_pickle='TRUE').item()
    cars = len(canonical_points)
    all_points = np.array([[0, 0, 0]])
    for i in range(cars):
        try:
            all_points = np.concatenate((all_points, canonical_points[i]))
            # if canonical_points[i].shape[0] > 200:
            #     obj = optimizer.reconstruct_object(np.array([[1, 0, 0, 0],
            #                                             [0, 1, 0, 0],
            #                                             [0, 0, 1, 0],
            #                                             [0, 0, 0, 1],
            #                                             ], dtype="float32"), canonical_points[i])
            #     objects_recon += [obj]    
        except KeyError:
            pass
    scale_pam = 1
    scale = scale_pam * np.array([
        [scale_pam, 0, 0],
        [0, scale_pam, 0,],
        [0, 0, scale_pam],
    ])
    all_points = (scale @ all_points.T).T
    rot_points = (rott[:3, :3] @ all_points.T).T
     
    obj = optimizer.reconstruct_object(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1],
                                            ], dtype="float32"), rot_points)
    objects_recon = [obj]

    # print("all_points", all_points)

    end = get_time()
    # print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_ctr = vis.get_view_control()

    # Add LiDAR point cloud
    # velo_pts, colors = kitti_seq.current_frame.get_colored_pts()
    scene_pcd = o3d.geometry.PointCloud()
    # scene_pcd.colors = o3d.utility.Vector3dVector(np.array([128, 0, 0]) / 255.0)


    c = 0
    scene_pcd.points = o3d.utility.Vector3dVector(all_points)
    vis.add_geometry(scene_pcd)
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)


    for i, obj in enumerate(objects_recon):
        # try:
        #     scene_pcd.points = o3d.utility.Vector3dVector(canonical_points[i])
        #     # print(f"canonical_points{i}", canonical_points[i].shape[0])
        #     if canonical_points[i].shape[0] > 200:
        #         vis.add_geometry(scene_pcd)
        #         c = c + canonical_points[i].shape[0]
                
        # except KeyError:
        #     pass

        mesh = mesh_extractor.extract_mesh_from_code(obj.code)

        # # write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(save_dir, "%d.ply" % obj_id))
        # write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join("./mesh", "%d.ply" % i))
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
 
        mesh_o3d.compute_vertex_normals()
        # mesh_o3d.paint_uniform_color(color_table[i])
        # Transform mesh from object to world coordinate
        # mesh_o3d.transform(obj.t_cam_obj)
        mesh_o3d.transform(rott)
        vis.add_geometry(mesh_o3d)

    # must be put after adding geometries
    # set_view(vis, dist=20, theta=0.)

    print("number of points cloud", c)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()


