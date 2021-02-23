
import time
from time import time


def center_point_face(faces):

    """
    Get center point of a prediction
    
    """
    points = []
    for face in faces:
        x, y, w, h = face
        
        points.append(round(x + w / 2, 2))
        points.append(round(y + h / 2, 2))
            
    return points


def predit_FaceDetector(images, model):

    """
    Measure times and point of a list predictions
    
    default threshold of model is 0.8
    
    posibles models:
    
    RfcnResnet101FaceDetector
    SSDMobileNetV1FaceDetector
    FasterRCNNFaceDetector
    YOLOv2FaceDetector
    TinyYOLOFaceDetector

    """

    predict_faces = []
    times = []
    
    for i in range(len(images)):
            
        start_time = time()
            
        faces = model.detect(images[i], include_score=False, draw_faces=False)

        total_time = time() - start_time
        
        predict_faces.append(center_point_face(faces))
        times.append(total_time)

    return predict_faces, times