
# Drone Battery Detection and 6D Pose Estimation
This project aims to utilize computer vision and machine learning techniques to detect and estimate the 6D pose of a specific type of drone battery，  
whcih could be applied on automatic drone battery hot-swapping. 

## Project Objectives:
Detect the battery in images
Estimate the 6D pose of the battery
Provide precise positioning for robotic manipulator grasping

## Installation:

1. Clone this repository:  
git clone https://github.com/Jiahao-Geng/drone-battery-detection-localization.git

2. Install dependencies:
pip install -r requirements.txt

## Usage

1. Prepare the dataset:
   - Place the images in `data/images/`
   - Place the labels in `data/labels/`
   - Create a `data/battery.yaml` file with dataset configuration

2. Train the model:
python src/train.py

3. Run inference:
python src/utils.py

## Dataset

This project uses a custom dataset containing images of specific battery types. The dataset includes:
- 300+ annotated images of 3D printed model batteries
- Various angles and lighting conditions
- 

## Model

using a modified version of YOLOv5 for object detection, combined with the PnP algorithm for pose estimation.

## Results

[Add some result images or video links here]
