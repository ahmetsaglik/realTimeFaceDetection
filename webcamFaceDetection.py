import cv2

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(1):
    _,frame = video.read()

    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayFrame,1.3,7)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)



    

    cv2.imshow("Video",frame)



    if cv2.waitKey(5) & 0xFF == ord('q'):
        break



video.release()
cv2.destroyAllWindows()
