import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

vid = cv2.VideoCapture(0)

while(1):
    ret,frame = vid.read()
    frame = cv2.flip(frame,1)
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayFrame,1.3,1)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    roi_frame = frame[y:y+h,x:x+w]
    roi_gray = grayFrame[y:y+h,x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)



    cv2.imshow("Video",frame)


    if cv2.waitKey(5) & 0xFF == ord('q'):
        break



vid.release()
cv2.destroyAllWindows()
