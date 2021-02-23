import cv2
import time
from time import time
from imutils import face_utils
import dlib


def img_detection(img, index, faces):

    """
    Draw a prediction with landmarks
    
    """
    
    for (i, rect) in enumerate(faces):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
    
    filename = './prediction/' + str(index) + '_test.jpg'
    cv2.imwrite(filename, img)


def center_point_face(faces):

    """
    Calculates center point of prediction
    
    """
    
    points = []
    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        
        points.append(round(x + w / 2, 2))
        points.append(round(y + h / 2, 2))
            
    return points


def predit_HOG(grays):

    """
    Measure times and points of list prediction

    for make predictions the images must be in gray
    
    """
    
    predict_faces = []
    times_HOG = []
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
    
    for i in range(len(grays)):
            
        start_time = time()
        faces = dnnFaceDetector(grays[i], 1)
        total_time = time() - start_time

        predict_faces.append(center_point_face(faces))
        times_HOG.append(total_time)
        
    return predict_faces, times_HOG
    
    
    

