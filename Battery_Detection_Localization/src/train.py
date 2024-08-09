import torch
from yolov5 import train


def train_model(data_yaml, weights='yolov5s.pt', epochs=100, img_size=640, batch_size=16):
    """
    Train the YOLOv5 model on the battery dataset.

    :param data_yaml: Path to the data.yaml file
    :param weights: Initial weights to use
    :param epochs: Number of epochs to train
    :param img_size: Image size for training
    :param batch_size: Batch size for training
    """
    train.run(data=data_yaml,
              weights=weights,
              epochs=epochs,
              img_size=img_size,
              batch_size=batch_size,
              project='runs/train',
              name='battery_detector')


if __name__ == '__main__':
    train_model('data/battery.yaml')