import cv2
import torch
import numpy as np

path = './dataset/best.pt'

plate = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    results=plate(frame)
    frame=np.squeeze(results.render())  
    cv2.imshow("TESTING",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()