# Drone Battery Detection and Pose Estimation  

This project aims to utilize computer vision and machine learning techniques to detect and estimate the 6D pose of a specific type of drone battery，  
whcih could be applied on automatic drone battery hot-swapping. 

## Project Objectives:  

Detect the battery in images
Estimate the 6D pose of the battery
Provide precise positioning for robotic manipulator grasping



# Usage Guide  

This guide provides step-by-step instructions for setting up and running the Battery Detection and Pose Estimation project using YOLOv5 and Intel RealSense camera.


## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (for training)
- Intel RealSense camera

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/battery-detection-project.git
   cd battery-detection-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

1. Collect images of batteries in various positions, angles, and lighting conditions.

2. Annotate the images:
   - Use a tool like LabelImg or CVAT to create bounding box annotations.
   - Save annotations in YOLO format (one .txt file per image).

3. Data Organizing :
   ```
   data/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── battery.yaml
   ```

4. Create `battery.yaml` file:
   ```yaml
   path: ../data
   train: images/train
   val: images/val

   nc: 1  # number of classes
   names: ['battery']
   ```

## Camera Calibration

1. Prepare a chessboard pattern for calibration.

2. Capture multiple images of the chessboard using the RealSense camera.

3. Run the calibration script:
   ```
   python src/camera_calibration.py
   ```

4. The calibration results will be saved in `camera_calibration.npz`.

## Model Training

1. Download pre-trained YOLOv5 weights:
   ```
   wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
   ```

2. Start training:
   ```
   python src/train.py --img 640 --batch 16 --epochs 100 --data data/battery.yaml --weights yolov5s.pt
   ```

3. Monitor training progress in the console and in the `runs/train/` directory.

## Running Detection

1. Ensure your RealSense camera is connected.

2. Run the detection script:
   ```
   python src/detect.py
   ```

3. The script will display real-time detection results. Press 'q' to quit.

## Validation

1. Prepare ground truth data:
   - Create a `ground_truth.json` file with the actual bounding boxes for your test scenario.

2. Run detection as described in the previous section.

3. After quitting the detection script, results and metrics will be saved in `final_detection_results.json`.

4. Review the metrics printed in the console and saved in the JSON file.
