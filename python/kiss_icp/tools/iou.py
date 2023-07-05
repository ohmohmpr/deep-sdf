# This is the format of the bounding box representation in OpenPCDet:
# 8.7975,  -7.5174,  -0.6397,   0.5892,   0.7885,   1.7722,   1.7198
# Where 
# 1. 0,1,2 are the x,y,z coordinates of the box center, 
# 2. 3,4,5 are the length, width, height of the box, and 
# 3. 6 is the yaw angle.


'''
Given box 1 and box 2, this file helps calculate the IoU of the two boxes.
box 1 and box 2 are in the format of [x,y,z,l,w,h,yaw]

SYNTAX:
iou = calculate_iou(box1, box2) # returns the IoU of box1 and box2
percent_overlap1, percent_overlap2 = calculate_percent_overlap(box1, box2) # returns the percent overlap of box1 and box2
'''

import numpy as np
import math
from scipy.spatial import ConvexHull


def find_intersection(segment1, segment2):

    Ax, Ay = segment1[0]
    Bx, By = segment1[1]
    Cx, Cy = segment2[0]
    Dx, Dy = segment2[1]

    # Calculate the slopes of the line segments
    m1 = (By - Ay) / (Bx - Ax) if Bx - Ax != 0 else float('inf')
    m2 = (Dy - Cy) / (Dx - Cx) if Dx - Cx != 0 else float('inf')

    # Check if the line segments are parallel or collinear
    if m1 == m2:
        return None  # Line segments are parallel or collinear, no intersection

    # Calculate the y-intercepts of the line segments
    b1 = Ay - m1 * Ax
    b2 = Cy - m2 * Cx

    # Calculate the x-coordinate of the intersection point
    x = (b2 - b1) / (m1 - m2)

    # Calculate the y-coordinate of the intersection point
    y = m1 * x + b1

    # Check if the intersection point lies within the bounds of both line segments
    if (
        min(Ax, Bx) <= x <= max(Ax, Bx) and
        min(Ay, By) <= y <= max(Ay, By) and
        min(Cx, Dx) <= x <= max(Cx, Dx) and
        min(Cy, Dy) <= y <= max(Cy, Dy)
    ):
        return (x, y)  # Line segments intersect

    return None  # Line segments do not intersect

def transform_line_segment(line_segment, traslation, rotation):
    # line segment example: np.array([[0, 0], [1, 1]])
    # create a homogeneous representation of the line segment
    line_segment = np.concatenate((line_segment, np.ones((2, 1))), axis=1).T
    # create the transformation matrix
    transformation_matrix = create_rt_joint_matrix(rotation, traslation)
    # apply the transformation
    transformed_line_segment = np.dot(transformation_matrix, line_segment)
    # return the transformed line segment
    return transformed_line_segment[:2].T

def create_rt_joint_matrix(rotation, translation):  
    return np.array([
        [np.cos(rotation), -np.sin(rotation), translation[0]],
        [np.sin(rotation), np.cos(rotation), translation[1]],
        [0, 0, 1]
    ])

# Given a box, compute all the line segments of the box
def compute_box_line_segments(box):
    # Box example: 8.7975,  -7.5174,  -0.6397,   0.5892,   0.7885,   1.7722,   1.7198 
    # Where 0,1,2 are the x,y,z coordinates of the box center, 3,4,5 are the length, width, height of the box, and 6 is the yaw angle.
    # Only consider the yaw angle for now
    yaw = box[6]
    # To make yaw non-zero, we need to rotate the to non-zero yaw
    if yaw == 0 or yaw == math.pi/2 or yaw == math.pi or yaw == 3*math.pi/2:
        yaw += 0.0001
    # Only consider 2D for now
    box_center = box[:2]
    box_length = box[3]
    box_width = box[4]
    
    # Compute the four corners of the box in "CANONICAL" frame
    top_left = np.array([-box_length / 2, box_width / 2])
    top_right = np.array([box_length / 2, box_width / 2])
    bottom_left = np.array([-box_length / 2, -box_width / 2])
    bottom_right = np.array([box_length / 2, -box_width / 2])

    # Line segments of the box in "CANONICAL" frame
    box_line_segments = np.array([ [top_left, top_right], [top_right, bottom_right], [bottom_right, bottom_left], [bottom_left, top_left] ])
    '''
    Visually, the box looks like this:

    CORNERS:

    top_left__________top_right
    |                    |
    bottom_left_______bottom_right

    LINE SEGMENTS:
            line1
    line4           line2
            line3
    
    '''
    for i in range(len(box_line_segments)):
        box_line_segments[i] = transform_line_segment(box_line_segments[i], box_center, yaw)
    
    return box_line_segments


# Compute intersection between line segment and box
def compute_line_segment_box_intersection(line_segment, box):
    # line segment example: np.array([[0, 0], [1, 1]])
    # box example: np.array([0, 0, 0, 1, 1, 1, 0])
    polygon_collection = []
    box_line_segments = compute_box_line_segments(box)
    for i in range(len(box_line_segments)):
        intersection = find_intersection(line_segment, box_line_segments[i])
        if intersection:
            polygon_collection.append(intersection)
    return polygon_collection

# Compute intersection between box1, box2
def compute_box_box_intersection(box1, box2):
    # Use line box intersection to compute the intersection between two boxes
    box1_line_segments = compute_box_line_segments(box1)
    box2_line_segments = compute_box_line_segments(box2)
    polygon_collection = []
    for i in range(len(box1_line_segments)):
        for j in range(len(box2_line_segments)):
            intersection = find_intersection(box1_line_segments[i], box2_line_segments[j])
            if intersection:
                polygon_collection.append(intersection)
    return polygon_collection

def boosted_polygon_collection(box1, box2):
    polygon_collection = compute_box_box_intersection(box1, box2)
    # Add points that are on both boxes from the vertices of the boxes
    pass
    ## TODO ##  

def point_in_box(point,box):
    # Check if the point is in the box
    pass
    ## TODO ##




# Calulate the area of the intersection between box1 and box2
# Using the shoelace formula

def calculate_polygon_area(polygon_collection):
    n = len(polygon_collection)
    if n == 0:
        return 0
    area = 0

    for i in range(n):
        x1, y1 = polygon_collection[i]
        x2, y2 = polygon_collection[(i + 1) % n]  # Wrap around to the first vertex for the last edge
        area += x1 * y2 - x2 * y1

    area /= 2

    return abs(area)

def calculate_common_area(box1,box2):
    polygon_collection = polygon_of_intersection(box1, box2)
    return polygon_collection, calculate_polygon_area(polygon_collection)

def calculate_iou(box1, box2):
    '''
    This function calculates the IoU of two 3D bounding boxes.
    Format of bounding box is [x,y,z,l,w,h,yaw]

    Input:
        box1: first bounding box in global frame
        box2: second bounding box in global frame
    Output:
        iou: IoU of the two boxes
    '''
    # Comvert the boxes to the numpy array format
    box1 = np.array(box1)
    box2 = np.array(box2)
    area_box1 = box1[3] * box1[4]
    area_box2 = box2[3] * box2[4]

    # Assumption: Roll and pitch angles are zero
    # Calculate common volume/area
    common_area = calculate_common_area(box1, box2)[1]
    # Calculate Union volume/area
    union_area = area_box1 + area_box2 - common_area
    # Calculate IoU
    iou = common_area / union_area

    return iou 

def calculate_percent_overlap(box1, box2):
    '''
    This function calculates the percent overlap of two 3D bounding boxes.
    Format of bounding box is [x,y,z,l,w,h,yaw]

    Input:
        box1: first bounding box in global frame
        box2: second bounding box in global frame
    Output:
        percent_overlap: percent overlap of the two boxes
    '''
    # Comvert the boxes to the numpy array format
    box1 = np.array(box1)
    box2 = np.array(box2)
    area_box1 = box1[3] * box1[4]
    area_box2 = box2[3] * box2[4]

    # Assumption: Roll and pitch angles are zero
    # Calculate common volume/area
    common_area = calculate_common_area(box1, box2)[1]
    # Calculate percent overlap
    percent_overlap1 = common_area / area_box1
    percent_overlap2 = common_area / area_box2
    return percent_overlap1, percent_overlap2

def is_point_inside_box(point, box):
    # Extracting box parameters
    box_center_x, box_center_y = box[0], box[1]
    box_length, box_width = box[3], box[4]
    box_yaw = box[6]

    # Extracting point coordinates
    point_x, point_y = point[0], point[1]

    # Shifting point to box-local coordinate system
    cos_yaw = math.cos(box_yaw)
    sin_yaw = math.sin(box_yaw)
    local_point_x = (point_x - box_center_x) * cos_yaw + (point_y - box_center_y) * sin_yaw
    local_point_y = -(point_x - box_center_x) * sin_yaw + (point_y - box_center_y) * cos_yaw

    # Checking if point is inside the box
    is_inside_x = abs(local_point_x) <= box_length / 2
    is_inside_y = abs(local_point_y) <= box_width / 2

    return is_inside_x and is_inside_y


def polygon_of_intersection(box1, box2):
    polygon_collection = compute_box_box_intersection(box1, box2)
    # Add points that are on both boxes from the vertices of the boxes
    box1_line_segments = compute_box_line_segments(box1)
    box2_line_segments = compute_box_line_segments(box2)
    for i in range(len(box1_line_segments)):
        if is_point_inside_box(box1_line_segments[i][0], box2):
            polygon_collection.append(box1_line_segments[i][0])

        # if is_point_inside_box(box1_line_segments[i][1], box2):
        #     polygon_collection.append(box1_line_segments[i][1])

    for i in range(len(box2_line_segments)):
        if is_point_inside_box(box2_line_segments[i][0], box1):
            polygon_collection.append(box2_line_segments[i][0])

        # if is_point_inside_box(box2_line_segments[i][1], box1):
        #     polygon_collection.append(box2_line_segments[i][1])

    if len(polygon_collection) <= 3:
        return []

    hull = ConvexHull(polygon_collection)
    ordered_vertices = [polygon_collection[i] for i in hull.vertices]    

    return ordered_vertices