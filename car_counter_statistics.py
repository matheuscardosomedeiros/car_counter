from ultralytics import YOLO
import cvzone
import cv2
import math
import numpy as np
import torch
from sort import *

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = YOLO('yolov8n.pt', task='detect')
model.to(device=device)

# Classes do modelo YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
              "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
              "toothbrush"]

mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [780, 587, 730, 716]
totalCount = 0
counted_ids = []

# Caminho para o vídeo local
video_path = "sample\Sábado 30-11-2024.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame, mask)

    if not ret:
        break  # Final do vídeo

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]

            # Verificar se é um carro
            if label == "car" and conf > 0.3:
                # Desenhar a caixa
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        print(result)
        cvzone.cornerRect(frame, (x1 ,y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=3, offset=10)
        
        cx,cy = x1+w//2, y1+h//2
        cv2.circle(frame,(cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[1]-5 < cy < limits[3]+5 and id not in counted_ids:
            totalCount += 1
            counted_ids.append(id)
            cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    
    cvzone.putTextRect(frame, f'Count: {totalCount}', (50, 50))
    print(f"*Contagem: {totalCount}")

    # Mostrar o vídeo
    cv2.imshow("Video", frame)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
