#!/usr/bin/env python3
import insightface
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import retinaface_r50_v1
model = retinaface_r50_v1()

def img_detection(faces, img):
    
    """
    Draw a predition with landmarks
    
    """
    
    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            #color = (255,0,0)   # changes color landmark
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

        cv2.imwrite("./output.jpg", img)


img = cv2.imread('img.jpg')

model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id=0, nms=0.3)

faces, landmark = model.detect(img, threshold=0.7, scale=1.0)

img_detection(faces, img)