import insightface
import cv2
import numpy as np
import time
from time import time


def img_detection(faces, index, img):
    
    """
    Draw a predition with landmarks
    
    """
    
    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            #color = (255,0,0)   # changes color landmark
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

        filename = './prediction/' + str(index) + '_test.jpg'
        print('writing', filename)
        cv2.imwrite(filename, img)
        
        
def center_point_face(faces):

    """
    Get center point of a prediction
    
    """
    points = []
    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            points.append(round((box[0] + box[2]) / 2, 2))
            points.append(round((box[1] + box[3]) / 2, 2))
            
    return points


def predit_retina_face(images, threshold, ctx_id=-1):

    """
    Measure times and points of list predictions
    
    ctx_id is for computacion on GPU minimun 6GB of vRAM
    
    """
    
    predict_faces = []
    times_retina_face = []
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=ctx_id, nms=0.3)
    
    for i in range(len(images)):

        start_time = time()
            
        faces, landmark = model.detect(images[i], threshold=threshold, scale=1.0)

        total_time = time() - start_time
        predict_faces.append(center_point_face(faces))
        times_retina_face.append(total_time)

    return predict_faces, times_retina_face




