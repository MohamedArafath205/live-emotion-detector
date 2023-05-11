import cv2
from deepface import DeepFace

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 4)
    name = DeepFace.analyze(img, actions=['emotion'])
    
    for (x,y,w,h) in faces:
        cv2.putText(img, name[0]['dominant_emotion'], (x,h + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2 )
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
    
    cv2.imshow('frame', img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()