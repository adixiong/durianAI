import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox

import cv2
import torch
import numpy as np

from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_model(device,weights,imgsz):
        # device= 'cuda:0' #for cpu-> 'cpu'
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride

    if half:
        model.half()  # to FP16

    names = model.module.names if hasattr(model, 'module') else model.names

    ###### infrence ####
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

    return stride, model, names, half ,old_img_w,old_img_h, old_img_b

def img_preprocessing(frame, imgsz, stride, device, half):
    img = letterbox(frame, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img

imgsz = 640
conf_thres = 0.70 #### threshold value
iou_thres = 0.45

weights = "runs/train/yolov7-custom/weights/best.pt"
imgsz = 640
conf_thres = 0.1
iou_thres = 0.45
device = select_device('cpu')### for gpu put 0
print(device)
### load yolov7 model ###
stride, model, names, half, old_img_w,old_img_h, old_img_b = load_model(device,weights,imgsz)

classes=['sultan_king_durian','musang_king_durian','red_prawn_durian']
colors=[(0,255,255),(255,255,0),(255,51,255)]

def model_predictor(frame):
    alpha=0.3
    global old_img_b, old_img_h, old_img_w
    tl = 3 or round(0.002 * (img.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)


    img = img_preprocessing(frame, imgsz, stride, device, half)

    ### Prediction
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img)[0]

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False) #Apply NMS
    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            bboxes=[]
            pred_classes=[]
            pred_scores=[]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                pred_class = (int(cls))
                pred_scores = round(float(conf), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), colors[pred_class], tl, lineType=cv2.LINE_AA)
                cv2.putText(frame,classes[pred_class],(x1,y1-2),0,tl / 3,(255,255,255),thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(frame,str(pred_scores),(x2,y1-2),0,tl / 3,(255,255,255),thickness=tf, lineType=cv2.LINE_AA)
    return frame


test_video_path="testing_video.mp4"

cap = cv2.VideoCapture(test_video_path)

if not cap.isOpened():
    print("Video path error.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No video stream check your video path............")
        break
    pred_img=model_predictor(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
