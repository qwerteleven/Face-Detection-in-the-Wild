import cv2
import matplotlib.pyplot as plt

from hof.face_detectors import RfcnResnet101FaceDetector, SSDMobileNetV1FaceDetector, FasterRCNNFaceDetector, \
    YOLOv2FaceDetector, TinyYOLOFaceDetector

MIN_CONFIDENCE = 0.5


def display(img, figsize=(15, 15)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


sample = cv2.imread("./img.jpg")

rfcn_face_detector = RfcnResnet101FaceDetector(min_confidence=MIN_CONFIDENCE)
rfcn_face_detector.detect(sample, color=(255, 0, 0), min_confidence=0.8)
display(sample)

faster_rcnn_face_detector = FasterRCNNFaceDetector(min_confidence=MIN_CONFIDENCE)
faster_rcnn_face_detector.detect(sample, color=(0, 255, 0))
display(sample)

ssd_face_detector = SSDMobileNetV1FaceDetector(min_confidence=MIN_CONFIDENCE)
ssd_face_detector.detect(sample, color=(0, 0, 255))
display(sample)

yolo_face_detector = YOLOv2FaceDetector(min_confidence=MIN_CONFIDENCE)
yolo_face_detector.detect(sample, color=(15, 235, 250))
display(sample)

yolo_face_detector = TinyYOLOFaceDetector(min_confidence=MIN_CONFIDENCE)
yolo_face_detector.detect(sample, color=(255, 255, 153))
display(sample)
