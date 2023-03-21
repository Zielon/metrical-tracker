# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS


# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

def keypoints_to_array(keypoint_list):
    return np.unique(np.hstack(([np.array(kp_ids) for kp_ids in keypoint_list])))


def merge_keypoint_ids(keypoint_lists):
    return np.hstack([keypoints_to_array(keypoint_list) for keypoint_list in keypoint_lists])


CONTOUR_LANDMARK_IDS = keypoints_to_array(FACEMESH_FACE_OVAL)
LEFT_EYEBROW_LANDMARK_IDS = keypoints_to_array(FACEMESH_LEFT_EYEBROW)
RIGHT_EYEBROW_LANDMARK_IDS = keypoints_to_array(FACEMESH_RIGHT_EYEBROW)
LEFT_EYE_LANDMARK_IDS = keypoints_to_array(FACEMESH_LEFT_EYE)
RIGHT_EYE_LANDMARK_IDS = keypoints_to_array(FACEMESH_RIGHT_EYE)
NOSE_LANDMARK_IDS = np.array([168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358])
LIPS_LANDMARK_IDS = keypoints_to_array(FACEMESH_LIPS)
MP_LANDMARKS = np.hstack((LEFT_EYEBROW_LANDMARK_IDS, RIGHT_EYEBROW_LANDMARK_IDS, LEFT_EYE_LANDMARK_IDS, RIGHT_EYE_LANDMARK_IDS, NOSE_LANDMARK_IDS, LIPS_LANDMARK_IDS))

LEFT_IRIS_LANDMARK_IDS = keypoints_to_array(FACEMESH_LEFT_IRIS)
RIGHT_IRIS_LANDMARK_IDS = keypoints_to_array(FACEMESH_RIGHT_IRIS)


# MP_LANDMARKS = np.hstack((LEFT_EYEBROW_LANDMARK_IDS, RIGHT_EYEBROW_LANDMARK_IDS, LEFT_EYE_LANDMARK_IDS, RIGHT_EYE_LANDMARK_IDS, NOSE_LANDMARK_IDS, LIPS_LANDMARK_IDS, LEFT_IRIS_LANDMARK_IDS, RIGHT_IRIS_LANDMARK_IDS))


def get_idx(index):
    idx = []
    for i, j in enumerate(MP_LANDMARKS):
        if j in index:
            idx.append(i)
    return idx
