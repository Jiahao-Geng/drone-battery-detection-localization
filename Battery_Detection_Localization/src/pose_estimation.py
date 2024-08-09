import cv2
import numpy as np


def estimate_pose(bbox, depth_frame, camera_matrix, dist_coeffs):
    """
    Estimate the 6D pose of a battery using PnP algorithm and depth information.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    width, height = x2 - x1, y2 - y1

    # Get depth at the center of the battery
    depth = depth_frame.get_distance(center_x, center_y)

    # 2D points in image plane
    image_points = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2], [center_x, center_y]
    ], dtype=np.float32)

    # 3D points in object coordinate system
    object_points = np.array([
        [-width / 2, -height / 2, 0],
        [width / 2, -height / 2, 0],
        [width / 2, height / 2, 0],
        [-width / 2, height / 2, 0],
        [0, 0, depth]
    ], dtype=np.float32)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    return success, rvec, tvec