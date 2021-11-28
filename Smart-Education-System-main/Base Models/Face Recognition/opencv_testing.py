import cv2
import face_recognition
import numpy as np
import urllib.request

from numpy.core.fromnumeric import resize

url='https://source.unsplash.com/cIEb4UJ4ruk'
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.resize(image,(0,0),None,0.35,0.35)

imgAnurag=face_recognition.load_image_file('D:\Smart-Education-System\Base Models/Face Data/Anurag Porel.jpg')
imgAnurag=cv2.cvtColor(imgAnurag,cv2.COLOR_BGR2RGB)
imgAnurag=cv2.resize(imgAnurag,(0,0),None,0.35,0.35)

faceLoc=face_recognition.face_locations(image)[0]
faceEncode=face_recognition.face_encodings(image)[0]
cv2.rectangle(image,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocA=face_recognition.face_locations(imgAnurag)[0]
encodeAnu=face_recognition.face_encodings(imgAnurag)[0]
cv2.rectangle(imgAnurag,(faceLocA[3],faceLocA[0]),(faceLocA[1],faceLocA[2]),(255,0,0),2)

result=face_recognition.compare_faces([faceEncode],encodeAnu,0.5)
dis=face_recognition.face_distance([faceEncode],encodeAnu)
print(result,dis)

cv2.imshow('Api',image)
cv2.imshow('Anurag',imgAnurag)
cv2.waitKey(0)