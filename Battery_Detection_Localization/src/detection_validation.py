import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / float(area1 + area2 - intersection)
    return iou


def validate_detections(detections, ground_truth, iou_threshold=0.5):
    """
    Validate detections against ground truth.

    :param detections: List of detection results
    :param ground_truth: List of ground truth bounding boxes
    :param iou_threshold: IoU threshold for considering a detection as correct
    :return: Dictionary containing precision, recall, and F1 score
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt_box in ground_truth:
        matched = False
        for detection in detections:
            if calculate_iou(gt_box, detection['bbox']) > iou_threshold:
                true_positives += 1
                matched = True
                break
        if not matched:
            false_negatives += 1

    false_positives = len(detections) - true_positives

    precision, recall, f1, _ = precision_recall_fscore_support(
        [1] * true_positives + [0] * false_positives,
        [1] * true_positives + [0] * false_negatives,
        average='binary'
    )

    return {'precision': precision, 'recall': recall, 'f1': f1}


def save_detection_results(detections, metrics, output_file):
    """
    Save detection results and metrics to a JSON file.

    :param detections: List of detections
    :param metrics: Dictionary containing precision, recall, and F1 score
    :param output_file: Path to the output JSON file
    """
    results = {
        'detections': detections,
        'metrics': metrics
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)