import torch
from yolov5 import detect
import cv2
import numpy as np
from utils import initialize_realsense, get_aligned_frames, draw_pose_axes
from pose_estimation import estimate_pose
from detection_validation import validate_detections, save_detection_results
import json

def detect_batteries(color_image, model):
    """
    Detect batteries in the color image using YOLOv5.
    """
    results = model(color_image)
    return results.pred[0]

def main():
    # Initialize RealSense
    pipeline = initialize_realsense()

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

    # Camera matrix and distortion coefficients (you need to calibrate your camera)
    camera_matrix = np.array([[615.0, 0, 320.0],
                              [0, 615.0, 240.0],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    # Load ground truth data
    with open('ground_truth.json', 'r') as f:
        ground_truth = json.load(f)

    final_detections = []

    try:
        while True:
            # Get aligned frames
            color_frame, depth_frame = get_aligned_frames(pipeline)
            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Detect batteries
            detections = detect_batteries(color_image, model)

            current_detections = []
            for *bbox, conf, cls in detections:
                if model.names[int(cls)] == 'battery':
                    # Estimate pose
                    success, rvec, tvec = estimate_pose(bbox, depth_frame, camera_matrix, dist_coeffs)

                    if success:
                        # Draw bounding box
                        cv2.rectangle(color_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                        # Draw pose axes
                        color_image = draw_pose_axes(color_image, rvec, tvec, camera_matrix, dist_coeffs)

                        # Display pose information
                        cv2.putText(color_image, f"X: {tvec[0][0]:.2f}", (int(bbox[0]), int(bbox[3])+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(color_image, f"Y: {tvec[1][0]:.2f}", (int(bbox[0]), int(bbox[3])+60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(color_image, f"Z: {tvec[2][0]:.2f}", (int(bbox[0]), int(bbox[3])+90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    current_detections.append({
                        'bbox': bbox,
                        'confidence': conf.item(),
                        'class': model.names[int(cls)],
                        'pose': {
                            'rvec': rvec.tolist() if success else None,
                            'tvec': tvec.tolist() if success else None
                        }
                    })

            # Store the current detections
            final_detections = current_detections

            # Display the result
            cv2.imshow("Battery Detection and Pose Estimation", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # Validate final detections
    metrics = validate_detections(final_detections, ground_truth)

    # Save final detections and metrics
    save_detection_results(final_detections, metrics, 'final_detection_results.json')

    print(f"Final Detection Results and Metrics saved to 'final_detection_results.json'")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()