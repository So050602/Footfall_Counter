#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-07T06:28:30.100Z
"""

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python-headless
!pip install ultralytics

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import torch
import cv2
import numpy as np
from collections import deque
from google.colab.patches import cv2_imshow
from ultralytics import YOLO # Import YOLO from ultralytics

# Load YOLOv5 model (person class: 0)
model = YOLO('yolov5s.pt') # Corrected: load YOLOv5s model directly
model.conf = 0.5  # Detection confidence threshold

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = dict()  # objectID -> centroid
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.objects.pop(objectID)
        self.disappeared.pop(objectID)

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(inputCentroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            for col in unusedCols:
                self.register(inputCentroids[col])
        return self.objects

# Input/Output paths -- adjust these as needed!
input_video_path = '/content/drive/MyDrive/Assignment_footfall/Test.mp4' # Check this path!
output_video_path = '/content/drive/MyDrive/Assignment_footfall/footfall_output.mp4'
roi_y = 1200  # Adjust: lower (closer to ground) for your frame!

cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read video")
height, width = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
tracker = CentroidTracker(maxDisappeared=80)

entry_count = 0
exit_count = 0
memory = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    # Access detections from the YOLO model results
    person_boxes = []
    for r in results:
        for *xyxy_tensor, conf_tensor, cls_tensor in r.boxes.data:
            if int(cls_tensor.cpu().numpy()) == 0 and conf_tensor.cpu().numpy() > 0.5: # Filter for 'person' (class 0)
                # Convert all tensor elements to CPU and then to numpy before appending
                x1, y1, x2, y2 = [t.cpu().numpy() for t in xyxy_tensor]
                conf = conf_tensor.cpu().numpy()
                cls = cls_tensor.cpu().numpy()
                person_boxes.append(np.array([x1, y1, x2, y2, conf, cls]))

    centroids = []
    for box in person_boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        centroids.append((cx, cy))

    objects = tracker.update(centroids)
    current_memory = {}

    # Draw yellow ROI
    cv2.line(frame, (0, roi_y), (width, roi_y), (0, 255, 255), 3)

    for object_id, centroid in objects.items():
        cv2.putText(frame, f"ID {object_id}", (centroid[0]-10, centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        current_memory[object_id] = centroid[1]

        if object_id in memory:
            prev_y = memory[object_id]
            curr_y = centroid[1]
            # Updated crossing logic
            if prev_y < roi_y <= curr_y: # Crossed from above ROI to below or on ROI
                exit_count += 1
            elif prev_y > roi_y >= curr_y: # Crossed from below ROI to above or on ROI
                entry_count += 1

    memory = current_memory.copy()

    cv2.putText(frame, f"Entries: {entry_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2_imshow(frame)

cap.release()
out.release()
print(f"Processed video at: {output_video_path}")