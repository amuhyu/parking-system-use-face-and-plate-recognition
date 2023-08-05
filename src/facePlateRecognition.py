import cv2
import torch
import numpy as np
import easyocr
import csv

# Load YOLOv5 model
path = './dataset/best.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Initialize the OCR reader
import easyocr
reader = easyocr.Reader(['en'], gpu=False)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

def detect_and_crop_webcam(confidence_threshold=0.5):
    cap = cv2.VideoCapture(0)  # Menggunakan webcam dengan nomor indeks 0

    csv_file = open('./results/ocr_results.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Label', 'OCR Text'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, size=640)
        detected_objects = results.pandas().xyxy[0]

        faces = facedetect.detectMultiScale(img_rgb, 1.1, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

        # Crop and display detected objects
        for _, obj in detected_objects.iterrows():
            label = obj['name']
            conf = obj['confidence']
            if conf > confidence_threshold:
                x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
                cropped_img = frame[y1:y2, x1:x2]

                # Perform OCR on cropped image
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
                # results = reader.readtext(gray_img, detail = 0, paragraph = True)
                results = reader.readtext(blur_img, detail = 0, paragraph= True)
                print(results)

                cv2.imshow(label, cropped_img)

                csv_writer.writerow([label, results])
        
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Tekan 'Esc' untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    detect_and_crop_webcam()
