import cv2 

video=cv2.VideoCapture(1)
facedetect = cv2.CascadeClassifier(".dataset/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Yusran", "Atta", "Awal", "Arif"]

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf>20 and conf <62:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 1)
            cv2.putText(frame, name_list[serial], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
            cv2.putText(frame, "Unknown", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            print(conf)
    frame=cv2.resize(frame, (640, 480))
    cv2.imshow("Frame",frame)
    
    k=cv2.waitKey(1)
    if k==ord("q"):
        break

video.release()
cv2.destroyAllWindows()