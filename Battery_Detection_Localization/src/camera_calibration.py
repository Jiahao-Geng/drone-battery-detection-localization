import numpy as np
import cv2
import glob


def calibrate_camera(chessboard_size, square_size, image_folder):
    """
    Calibrate the camera using chessboard images.

    :param chessboard_size: Tuple of (width, height) of the chessboard
    :param square_size: Size of each square in the chessboard (in meters)
    :param image_folder: Folder containing calibration images
    :return: camera_matrix, dist_coeffs
    """
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Get list of calibration images
    images = glob.glob(f'{image_folder}/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs


def save_calibration(camera_matrix, dist_coeffs, filename='camera_calibration.npz'):
    """
    Save camera calibration results to a file.

    :param camera_matrix: Camera matrix
    :param dist_coeffs: Distortion coefficients
    :param filename: Output filename
    """
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def load_calibration(filename='camera_calibration.npz'):
    """
    Load camera calibration results from a file.

    :param filename: Input filename
    :return: camera_matrix, dist_coeffs
    """
    with np.load(filename) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    # Example usage
    chessboard_size = (9, 6)  # Number of inner corners on the chessboard
    square_size = 0.025  # Size of each square in meters
    image_folder = 'calibration_images'  # Folder containing calibration images

    camera_matrix, dist_coeffs = calibrate_camera(chessboard_size, square_size, image_folder)
    save_calibration(camera_matrix, dist_coeffs)
    print("Camera calibration completed and saved.")