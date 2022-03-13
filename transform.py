from typing import Tuple, List
import sys
import math
import numpy as np


DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def get_camera_matrix(image_size: Tuple[int, int], focal_length: float) -> np.ndarray:
    return np.array([[image_size[0] * focal_length, 0, image_size[0] * 0.5],
                     [0, image_size[0] * focal_length, image_size[1] * 0.5],
                     [0, 0, 1]], float)


def get_pitch_rotation(pitch: float) -> np.ndarray:
    rad_pitch = pitch * DEG2RAD
    return np.matrix([[1, 0, 0],
                      [0, math.cos(rad_pitch), - math.sin(rad_pitch)],
                      [0, math.sin(rad_pitch), math.cos(rad_pitch)]])


def get_yaw_rotation(yaw: float) -> np.ndarray:
    rad_yaw = yaw * DEG2RAD
    return np.matrix([[math.cos(rad_yaw), 0, math.sin(rad_yaw)],
                      [0, 1, 0],
                      [- math.sin(rad_yaw), 0, math.cos(rad_yaw)]])


def get_roll_rotation(roll: float) -> np.ndarray:
    rad_roll = roll * DEG2RAD
    return np.matrix([[math.cos(rad_roll), - math.sin(rad_roll), 0],
                      [math.sin(rad_roll), math.cos(rad_roll), 0],
                      [0, 0, 1]])


def remove_z_axis(M: np.ndarray) -> np.ndarray:
    return np.stack([M[:3, 0], M[:3, 1], M[:3, 3]])


def get_rotation_matrix(angles: Tuple[float, float, float]) -> np.ndarray:
    r_x, r_y, r_z = get_pitch_rotation(angles[0]), get_yaw_rotation(angles[1]), get_roll_rotation(angles[2])
    return np.dot(r_z, np.dot(r_y, r_x))


def get_transform_matrix(angles: Tuple[float, float, float], translation: Tuple[float, float, float]) -> np.ndarray:
    transform_matrix = np.identity(4, float)
    transform_matrix[:3, :3] = get_rotation_matrix(angles)
    transform_matrix[:3, 3] = translation
    return transform_matrix


def get_look_transform(angles: Tuple[float, float, float], distance: float) -> np.ndarray:
    return get_transform_matrix(angles, (0, 0, distance))


def get_affine_fit_image_matrix(input_image_size: Tuple[int, int],
                                output_image_size: Tuple[int, int]) -> np.ndarray:
    scale = max(output_image_size[0] / input_image_size[0], output_image_size[1] / input_image_size[1])
    offset = ((output_image_size[0] - scale * input_image_size[0]) * 0.5,
              (output_image_size[1] - scale * input_image_size[1]) * 0.5)
    return np.array([[scale, 0, offset[0]],
                     [0, scale, offset[1]]], float)


def get_fit_transformed_image_matrix(input_image_size: Tuple[int, int],
                                     output_image_size: Tuple[int, int],
                                     transform_matrix: np.ndarray,
                                     extend_scale: float = 0.05) -> np.ndarray:
    image_corners = [np.dot(transform_matrix, np.array(p, float))
                     for p in [[0, 0, 1], [input_image_size[0], 0, 1],
                               [input_image_size[0], input_image_size[1], 1],  [0, input_image_size[1], 1]]]
    image_corners = [(v[0] / v[2], v[1] / v[2]) for v in image_corners]

    bb_min = [sys.float_info.max, sys.float_info.max]
    bb_max = [- sys.float_info.max, - sys.float_info.max]

    for corner in image_corners:
        if corner[0] < bb_min[0]:
            bb_min[0] = corner[0]
        if corner[1] < bb_min[1]:
            bb_min[1] = corner[1]
        if corner[0] > bb_max[0]:
            bb_max[0] = corner[0]
        if corner[1] > bb_max[1]:
            bb_max[1] = corner[1]

    scale = min(output_image_size[0] / (bb_max[0] - bb_min[0]), output_image_size[1] / (bb_max[1] - bb_min[1]))
    scale *= (1 - extend_scale)

    offset = (- bb_min[0] * scale, - bb_min[1] * scale)

    borders = (output_image_size[0] - (bb_max[0] - bb_min[0]) * scale,
               output_image_size[1] - (bb_max[1] - bb_min[1]) * scale)

    offset = (offset[0] + borders[0] * 0.5, offset[1] + borders[1] * 0.5)

    return np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]], float)


def normalize_image_corners(image_corners: List[Tuple[float, float]],
                            image_size: Tuple[int, int]) -> List[Tuple[float, float]]:
    return [(c[0] / image_size[0], c[1] / image_size[1]) for c in image_corners]


def denormalize_image_corners(normalized_image_corners: List[Tuple[float, float]],
                              image_size: Tuple[int, int]) -> List[Tuple[float, float]]:
    return [(c[0] * image_size[0], c[1] * image_size[1]) for c in normalized_image_corners]


def matrix_to_image_corners(matrix: np.ndarray, input_image_size: Tuple[int, int]) -> List[Tuple[float, float]]:
    corners = [np.dot(matrix, [c[0], c[1], 1]) for c in [(0, 0),
                                                         (input_image_size[0], 0),
                                                         (input_image_size[0], input_image_size[1]),
                                                         (0, input_image_size[1])]]
    return [(c[0] / c[2], c[1] / c[2]) for c in corners]


def image_corners_to_matrix(image_corners: List[Tuple[float, float]], input_image_size: Tuple[int, int]) -> np.ndarray:
    """
    This is a implementation of DLT-algorithm: getting the transformation matrix from input / output coordinates.
    :param image_corners: output image corners after the perspective transformation
    :param input_image_size: the size of the input image
    :return: calculated transformation
    """

    assert len(image_corners) == 4

    input_image_corners = [(0, 0), (input_image_size[0], 0),
                           (input_image_size[0], input_image_size[1]), (0, input_image_size[1])]

    matrix_a_data = []
    for output_image_corner, input_image_corner in zip(image_corners, input_image_corners):
        matrix_a_data.append([input_image_corner[0], input_image_corner[1], 1,
                              0, 0, 0,
                              - input_image_corner[0] * output_image_corner[0],
                              - input_image_corner[1] * output_image_corner[0],
                              - output_image_corner[0]])
        matrix_a_data.append([0, 0, 0,
                              input_image_corner[0], input_image_corner[1], 1,
                              - input_image_corner[0] * output_image_corner[1],
                              - input_image_corner[1] * output_image_corner[1],
                              - output_image_corner[1]])
    matrix_a_data.append([0, 0, 0, 0, 0, 0, 0, 0, 0])

    u, w, vt = np.linalg.svd(np.array(matrix_a_data, float), False, True)
    h_vec = vt[-1, :]
    h = np.array([[h_vec[0], h_vec[1], h_vec[2]],
                  [h_vec[3], h_vec[4], h_vec[5]],
                  [h_vec[6], h_vec[7], h_vec[8]]], float)
    h /= h[2, 2]
    return h

