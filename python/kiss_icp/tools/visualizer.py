# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import copy
import importlib
import os
from abc import ABC
from functools import partial
from typing import Callable, List

import numpy as np

import open3d as o3d
from bbox import  BBox3D
from bbox.metrics import iou_3d
from pyquaternion import Quaternion
from .iou import calculate_iou

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
BLUE = np.array([0.4, 0.5, 0.9])
SPHERE_SIZE = 0.20

global_view = False

class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, source, keypoints, target_map, pose):
        pass


class RegistrationVisualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(
        self,
        first_frame_bboxes,
    ):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create data
        self.source = self.o3d.geometry.PointCloud()
        self.keypoints = self.o3d.geometry.PointCloud()
        self.target = self.o3d.geometry.PointCloud()
        self.frames = []
        self.count = 0

        # data association
        self.first_frame_bboxes = first_frame_bboxes
        self.instances = [] 
        self.visual_instances = []
        self.prev_boxes = []
        self.visual_prev_boxes = []
        
        # Initialize visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

        # Visualization options
        self.render_map = True
        self.render_source = True
        self.render_keypoints = False
        self.global_view = global_view
        self.render_trajectory = True
        # Cache the state of the visualizer
        self.state = (
            self.render_map,
            self.render_keypoints,
            self.render_source,
        )
        
    def update(self, source, keypoints, target_map, pose, bboxes):
        target = target_map.point_cloud()
        self._update_geometries(source, keypoints, target, pose, bboxes)
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.source)
        self.vis.add_geometry(self.keypoints)
        self.vis.add_geometry(self.target)
        
        
        self.create_bboxes(self.first_frame_bboxes, np.array([[1, 0, 0, 0], 
                                                            [0, 1, 0, 0], 
                                                            [0, 0, 1, 0], 
                                                            [1, 0, 0, 1]]))
        
        self._set_black_background(self.vis)
        self.vis.get_render_option().point_size = 1
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/sdfdsstart\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [F] to toggle on/off the input cloud to the pipeline\n"
            "\t    [K] to toggle on/off the subsbampled frame\n"
            "\t    [M] to toggle on/off the local map\n"
            "\t    [V] to toggle ego/global viewpoint\n"
            "\t    [T] to toggle the trajectory view(only available in global view)\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)
        self._register_key_callback(["V"], self._toggle_view)
        self._register_key_callback(["C"], self._center_viewpoint)
        self._register_key_callback(["F"], self._toggle_source)
        self._register_key_callback(["K"], self._toggle_keypoints)
        self._register_key_callback(["M"], self._toggle_map)
        self._register_key_callback(["T"], self._toggle_trajectory)
        self._register_key_callback(["B"], self._set_black_background)
        self._register_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_source(self, vis):
        if self.render_keypoints:
            self.render_keypoints = False
            self.render_source = True
        else:
            self.render_source = not self.render_source
        return False

    def _toggle_keypoints(self, vis):
        if self.render_source:
            self.render_source = False
            self.render_keypoints = True
        else:
            self.render_keypoints = not self.render_keypoints

        return False

    def _toggle_map(self, vis):
        self.render_map = not self.render_map
        return False

    def _toggle_view(self, vis):
        self.global_view = not self.global_view
        self._trajectory_handle()

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _toggle_trajectory(self, vis):
        if not self.global_view:
            return False
        self.render_trajectory = not self.render_trajectory
        self._trajectory_handle()
        return False

    def _trajectory_handle(self):
        if self.render_trajectory and self.global_view:
            for frame in self.frames:
                self.vis.add_geometry(frame, reset_bounding_box=False)
        else:
            for frame in self.frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)

    def _update_geometries(self, source, keypoints, target, pose, bboxes):
        # Source hot frame
        if self.render_source:
            self.source.points = self.o3d.utility.Vector3dVector(source)
            self.source.paint_uniform_color(RED)
            if self.global_view:
                self.source.transform(pose)
        else:
            self.source.points = self.o3d.utility.Vector3dVector()

        # Keypoints
        if self.render_keypoints:
            self.keypoints.points = self.o3d.utility.Vector3dVector(keypoints)
            self.keypoints.paint_uniform_color(YELLOW)
            if self.global_view:
                self.keypoints.transform(pose)
        else:
            self.keypoints.points = self.o3d.utility.Vector3dVector()

        # Target Map
        if self.render_map:
            target = copy.deepcopy(target)
            self.target.points = self.o3d.utility.Vector3dVector(target)
            self.target.paint_uniform_color(BLUE)
            if not self.global_view:
                self.target.transform(np.linalg.inv(pose))
        else:
            self.target.points = self.o3d.utility.Vector3dVector()

        # Update always a list with all the trajectories
        new_frame = self.o3d.geometry.TriangleMesh.create_sphere(SPHERE_SIZE)
        new_frame.paint_uniform_color(BLUE)
        new_frame.compute_vertex_normals()
        new_frame.transform(pose)
        self.frames.append(new_frame)
        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.vis.add_geometry(self.frames[-1], reset_bounding_box=False)

        self.vis.update_geometry(self.keypoints)
        self.vis.update_geometry(self.source)
        self.vis.update_geometry(self.target)
        self.create_bboxes(bboxes, pose)
        
        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False



    def translate_boxes_to_open3d_instance(self, bbox):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        """
        center = bbox[0:3]
        lwh = bbox[3:6]
        axis_angles = np.array([0, 0, bbox[6] + 1e-10])
        rot = self.o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = self.o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = self.o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        return line_set, box3d
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d

    def create_bboxes(self, bboxes, pose):
        
        # print("#################### BEFORE ####################")
        # print("len(self.instances)", len(self.instances))
        # print("len(self.visual_instances)", len(self.visual_instances))
        # print("self.instances", self.instances)
        # print("#################### BEFORE ####################")
        
        self.prev_boxes = self.instances.copy() 
        self.visual_prev_boxes = self.visual_instances.copy() 
        
        num_found_instances =  0 # Remove not found objects
        num_total_instances = len(self.visual_prev_boxes) # Remove not found objects
        for i in range(bboxes.shape[0]):
            is_the_same_instance = False
            box = bboxes[i] #xyz lwh yaw in axis-angle
            
            # Create a box
            line_set, box3d = self.translate_boxes_to_open3d_instance(box)
            line_set.paint_uniform_color(np.random.rand(3)) # line_set.paint_uniform_color((0, 1, 0))
            
            if global_view:
                # box3d.transform(pose)
                box_t = np.hstack((box[0:3], np.array(1)))
                hom_pose = pose @ box_t
                box[0:3] = hom_pose[0:3]
                line_set.transform(pose)
            
            # Check IOU a new box and prev_boxes
            if len(self.instances) != 0:
                is_the_same_instance, idx_prev_boxes = self.test(box)
                # is_the_same_instance, idx_prev_boxes = self.test(box3d)
                # print("idx_prev_boxes", idx_prev_boxes)
            # print("self.count", self.count)
            
            if is_the_same_instance:
                num_found_instances = num_found_instances + 1
                
                # UPDATE GEOMETRY
                color = np.asarray(self.visual_prev_boxes[idx_prev_boxes].colors[0])
                # print("Color", color)
                
                line_set.paint_uniform_color(color)
                self.vis.remove_geometry(self.visual_prev_boxes[idx_prev_boxes], reset_bounding_box=False)

                self.prev_boxes.pop(idx_prev_boxes)
                self.visual_prev_boxes.pop(idx_prev_boxes)
    
            
            print("box", box[0:3])
            self.vis.add_geometry(line_set, reset_bounding_box=False)
            self.instances.append(box)
            self.visual_instances.append(line_set)
            
        self.count = self.count + 1
        # Remove not found objects
        num_not_found_instances = num_total_instances - num_found_instances
        self.remove_not_found_box(num_not_found_instances)
        
        

        # print("#################### AFTER ####################")
        
        num_of_instances_in_this_frame = bboxes.shape[0]
        self.instances = self.instances[-num_of_instances_in_this_frame:].copy()
        self.visual_instances = self.visual_instances[-num_of_instances_in_this_frame:].copy()
        
        # print("len(self.instances)", len(self.instances))
        # print("len(self.visual_instances)", len(self.visual_instances))
        # print("self.instances", self.instances)
        
        # print("#################### AFTER ####################")
        
        
    def remove_not_found_box(self, num_not_found_instances):
        for i in range(num_not_found_instances):
            self.vis.remove_geometry(self.visual_prev_boxes[i], reset_bounding_box=False)
                

            ############################################################
            # find the way to update, it might be better.
            # print("self.line_set", np.asarray(self.line_set.lines))
            # self.vis.update_geometry(self.line_set)
            ############################################################

    def vector3(self, x,y,z):
        return np.array((x,y,z), dtype=float)

    def Rotmat2Euler(self, rotmat ):
    # ############################################################################
    # Function computes the Euler Angles from given rotation matrix
    # ----------------------------------------------------------------------------
    # Input:
    # rotmat (double 3x3)...matrix
    #-----------------------------------------------------------------------------
    # @author: Felix Esser
    # @date: 14.07.2020
    # @mail: s7feesse@uni-bonn.de
    # @ literature: Förstner and Wrobel (2016), Photogrammetric Computer Vision
    # ############################################################################

        roll  = np.arctan2( rotmat[2,1], rotmat[2,2] )
        pitch = np.arctan2( -rotmat[2,0], np.sqrt(rotmat[2,1]**2 + rotmat[2,2]**2) )
        yaw   = np.arctan2( rotmat[1,0], rotmat[0,0] )

        rpy = self.vector3( roll, pitch, yaw )

        return rpy


    def test(self, box):
        num_prev_boxes = len(self.prev_boxes)
        # print("num_prev_boxes => ", num_prev_boxes)
        is_found = False
        idx_prev_boxes = -1
        for i in range(num_prev_boxes):
            
            prev_b = self.prev_boxes[i]
            current_b = box
            
            
            prev_b_axis_angles = np.array([0, 0, prev_b[6] + 1e-10])
            prev_b_rot = o3d.geometry.get_rotation_matrix_from_axis_angle(prev_b_axis_angles)
            yaw_1 = self.Rotmat2Euler(prev_b_rot)[2]
            prev_b[6] = yaw_1
            # prev_b_q8d = Quaternion(matrix=prev_b_rot)
            
            current_b_axis_angles = np.array([0, 0, current_b[6] + 1e-10])
            current_b_rot = o3d.geometry.get_rotation_matrix_from_axis_angle(current_b_axis_angles)
            yaw_2 = self.Rotmat2Euler(current_b_rot)[2]
            current_b[6] = yaw_2
            # current_b_q8d = Quaternion(matrix=current_b_rot)

            # box1 = BBox3D(*prev_b[:-1], prev_b_q8d)
            # box2 = BBox3D(*current_b[:-1], current_b_q8d)
        
            # if (iou_3d(box1, box2) > 0):
            #     is_found = True
            #     idx_prev_boxes = i
            #     break
            

            if (calculate_iou(prev_b, current_b) > 0):
                is_found = True
                idx_prev_boxes = i
                break
                
        return is_found, idx_prev_boxes