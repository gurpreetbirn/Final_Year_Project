from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 ## import openCV library
import numpy as np
import pkg_resources

haarFaceCascade = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')

faceDetectionClassifier = cv2.CascadeClassifier(haarFaceCascade)
FacialExpressionclassifier = load_model('C:/Users/gurpr/Desktop/ce301_birn_g/python_desktop_app/Trained_Emotion_VGG.h5')

emotionLables = ['Angry','Happy','Neutral','Sad','Surprise']


captureVideo = cv2.VideoCapture(0)##this will be accessing the webcam


while True:
    # Grab a single frame of video
    ret, frame = captureVideo.read()
    emotions = []##delete
    colourToGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faceDetected = faceDetectionClassifier.detectMultiScale(colourToGray,1.3,5)

    for (x,y,w,h) in faceDetected:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        roiGray = colourToGray[y:y+h,x:x+w]
        roiGray = cv2.resize(roiGray,(48,48),interpolation=cv2.INTER_AREA)
   
        if np.sum([roiGray])!=0:
            rectRegion = roiGray.astype('float')/255.0
            rectRegion = img_to_array(rectRegion)
            rectRegion = np.expand_dims(rectRegion,axis=0)

        # make a prediction on the rectRegion, then lookup the class

            predict = FacialExpressionclassifier.predict(rectRegion)[0]
            emotion= emotionLables[predict.argmax()]
            emotionLableLocation = (x,y)
            cv2.putText(frame,emotion,emotionLableLocation,cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        else:
            cv2.putText(frame,'Face Not in View',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    cv2.imshow('Facial emotion detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


captureVideo.release()
cv2.destroyAllWindows()

