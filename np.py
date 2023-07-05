from bbox import  BBox3D
from bbox.metrics import iou_3d
import numpy as np
from pyquaternion import Quaternion

import open3d as o3d

bounding_boxes_sq = np.load('sequences/sequence00_01.npy', allow_pickle='TRUE').item()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

transformation_matrix = np.load('transformation_matrix.npy', allow_pickle='TRUE').item()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

print(bounding_boxes_sq[0])
    

