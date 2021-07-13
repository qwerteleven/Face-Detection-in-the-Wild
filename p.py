import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread('img.jpg')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./output.jpg", rimg)
