import cv2
import numpy
import face_recognition

imgAnurag=face_recognition.load_image_file('D:\Smart-Education-System\Base Models/Face Data/Anurag Porel.jpg')
imgAnurag=cv2.cvtColor(imgAnurag,cv2.COLOR_BGR2RGB)
imgAnurag=cv2.resize(imgAnurag,(0,0),None,0.35,0.35)
imgAnuragT=face_recognition.load_image_file('D:\Smart-Education-System\Base Models/Face Data/Anurag Test3.jpg')
imgAnuragT=cv2.cvtColor(imgAnuragT,cv2.COLOR_BGR2RGB)
imgAnuragT=cv2.resize(imgAnuragT,(0,0),None,0.35,0.35)

faceLoc=face_recognition.face_locations(imgAnurag)[0]
encodeAnu=face_recognition.face_encodings(imgAnurag)[0]
cv2.rectangle(imgAnurag,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocT=face_recognition.face_locations(imgAnuragT)[2]
encodeAnuT=face_recognition.face_encodings(imgAnuragT)[2]
cv2.rectangle(imgAnuragT,(faceLocT[3],faceLocT[0]),(faceLocT[1],faceLocT[2]),(255,0,0),2)

results=face_recognition.compare_faces([encodeAnu],encodeAnuT,0.5)
faceDis=face_recognition.face_distance([encodeAnu],encodeAnuT)
print(results,faceDis)


cv2.imshow('Anurag',imgAnurag)
cv2.imshow('test',imgAnuragT)
cv2.waitKey(0)