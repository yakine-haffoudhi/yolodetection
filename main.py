import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
cap = cv2.VideoCapture("C:/Users/MSI/Downloads/BEST OF CHOUFLI HAL.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #YOLOv8 inference
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        class_ids = result.boxes.cls.numpy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f'{model.names[int(class_id)]} {score:.2f}'
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
