# Knee-X-Ray-Keypoints-Detection-with-YOLO11-Pose-Estimation-Training

This project uses YOLOv11 for keypoints detection and pose estimation on knee X-ray images. It processes input X-ray images, detects keypoints on the knees, and outputs annotated images showing the detected points.

## Requirements

To run this project, you will need to install the following Python packages:

- `flask`: Web framework to serve the app.
- `opencv-python`: For image processing tasks.
- `ultralytics`: YOLO model for object detection.
- `supervision`: For annotation and visualization.
- `pyresearch`: Custom package (make sure to install if needed).

You can install these dependencies via `pip`:

```bash
pip install flask opencv-python ultralytics supervision pyresearch
