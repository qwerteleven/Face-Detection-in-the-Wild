import insightface
import cv2
import numpy as np


def img_detection(faces, img):
    
    """
    Draw a predition with landmarks
    
    """
    
    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            #color = (255, 0, 0)   # changes color landmark
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

        return img
        
        
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


def predit_retina_face(model_name, img, threshold=0.3, ctx_id=-1, scale=1.0, nms=0.3):

    """
    Measure times and points of list predictions
    
    ctx_id is for computacion on GPU minimun 6GB of vRAM
    
    """

    model = insightface.model_zoo.get_model(model_name)
    model.prepare(ctx_id=ctx_id, nms=nms)
            
    faces, landmark = model.detect(img, threshold=threshold, scale=scale)

    return  img_detection(faces, img)


if __name__ == '__main__' :
    model_name = 'retinaface_r50_v1'
    img = cv2.imread('img.jpg')

    predit_retina_face(model_name, img)




