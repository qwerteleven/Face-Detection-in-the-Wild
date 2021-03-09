import cv2
import time
from time import time


def img_detection(img, index, faces):

    """
    Draw a prediction with landmarks
    
    """
    
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    filename = './prediction/' + str(index) + '_test.jpg'
    cv2.imwrite(filename, img)


def center_point_face(faces):

    """
    Calculates center point of a prediction
    
    """
    
    points = []
    for (x, y, w, h) in faces:
        points.append(round(x + w / 2, 2))
        points.append(round(y + h / 2, 2))
            
    return points


def predit_haar(grays, clasificator):

    """
    Measure times and points of list predictions
    
    posible clasificators:
    
    haarcascade_frontalface_default.xml
    haarcascade_profileface.xml
    lbpcascade_profileface.xml
    lbpcascade_frontalface.xml
    haarcascade_frontalface_alt.xml
    
    need it rute of file
    
    """
    
    predict_faces = []
    times_haar = []
    face_cascade = clasificator
    
    for i in range(len(grays)):
            
        start_time = time()
        faces = face_cascade.detectMultiScale(grays[i], 1.3, 5)
        total_time = time() - start_time

        predict_faces.append(center_point_face(faces))
        times_haar.append(total_time)
        
    return predict_faces, times_haar


