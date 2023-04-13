import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("E:/download/Cvzone/Videos/cars.mp4")

model = YOLO("/Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('CarCount/mask.png')
# print(mask.shape) --> (1080, 1080, 3)

# prev_frame_time = 0
# new_frame_time = 0

# Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []

while True:
    # new_frame_time = time.time()
    success, img = cap.read()
    # print(img.shape) --> (720, 1280, 3)

    # put mask to get aimed area
    imgRegion = cv2.bitwise_and(img,mask)

    # put decorate image
    imgGraphics = cv2.imread("CarCount/graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(0,0))

    results = model(imgRegion,stream=True)

    detections = np.empty((0,5))

    # print(results)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # print(box)
            """
            ultralytics.yolo.engine.results.Boxes object with attributes:

                boxes: tensor([[6.1884e+02, 1.1714e+02, 6.3634e+02, 1.3602e+02, 3.0743e-01, 2.0000e+00]], device='cuda:0')
                cls: tensor([2.], device='cuda:0')
                conf: tensor([0.3074], device='cuda:0')
                data: tensor([[6.1884e+02, 1.1714e+02, 6.3634e+02, 1.3602e+02, 3.0743e-01, 2.0000e+00]], device='cuda:0')
                id: None
                is_track: False
                orig_shape: tensor([ 720, 1280], device='cuda:0')
                shape: torch.Size([1, 6])
                xywh: tensor([[627.5885, 126.5779,  17.4949,  18.8760]], device='cuda:0')
                xywhn: tensor([[0.4903, 0.1758, 0.0137, 0.0262]], device='cuda:0')
                xyxy: tensor([[618.8411, 117.1399, 636.3359, 136.0160]], device='cuda:0')
                xyxyn: tensor([[0.4835, 0.1627, 0.4971, 0.1889]], device='cuda:0')
            """
            # get data
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1,y2-y1
            # draw 
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0] * 100))/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # selection aimed classes
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                # print(currentArray) [         32          33         193         107        0.91]
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # count numbers
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

            # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)

    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    # print(fps)

    cv2.imshow("Image",img)
    cv2.waitKey(1)


    






# cv2.imshow("mask",mask)
# cv2.waitKey(1)
