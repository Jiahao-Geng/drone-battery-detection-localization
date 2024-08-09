import pyrealsense2 as rs
import numpy as np
import cv2


def initialize_realsense():
    """
    Initialize the RealSense camera pipeline.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


def get_aligned_frames(pipeline):
    """
    Get aligned color and depth frames from the RealSense camera.
    """
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    return color_frame, depth_frame


def draw_pose_axes(image, rvec, tvec, camera_matrix, dist_coeffs, length=0.1):
    """
    Draw pose axes on the image.
    """
    axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    img_points = img_points.astype(int)
    origin = tuple(img_points[0].ravel())
    image = cv2.line(image, origin, tuple(img_points[1].ravel()), (0, 0, 255), 5)
    image = cv2.line(image, origin, tuple(img_points[2].ravel()), (0, 255, 0), 5)
    image = cv2.line(image, origin, tuple(img_points[3].ravel()), (255, 0, 0), 5)

    return image