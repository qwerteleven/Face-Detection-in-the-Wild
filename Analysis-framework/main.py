import cv2
import os
import math
import neural_networks.retinaFace_ as retinaFace_
import multiprocessing
import tradicional_methods.HOG as HOG
import neural_networks.faceDetectorHof as faceDetectorHof
import tradicional_methods.tradicional_methods as tradicional_methods
import matplotlib.pyplot as plt
from hof.face_detectors import RfcnResnet101FaceDetector, SSDMobileNetV1FaceDetector, FasterRCNNFaceDetector, \
    YOLOv2FaceDetector, TinyYOLOFaceDetector

predict = []
times = []
threshold = 0.3

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
path = "./dataset/"
path_labels = "./labels/face_labels.txt"


def load_labels(folder):
    """
    Load file with the detections and the name of image
    
    *see convert_YOLO_labels for create this file
    
    """

    files = []
    labels = []
    f = open(folder, "r")
    lines = f.readlines()
    for line in lines:
        values = line.split(" ")
        points = []
        for i in range(len(values)):
            if i == 0:
                files.append(values[i])
            else:
                if values[i] != "\n": points.append(float(values[i]))
        labels.append(points)
    f.close()

    return files, labels


def load_images_from_folder(folder):
    """
    Load dataset images and names in two version Gray and BGR
    
    *see cv2 imread()
    
    """

    images = []
    grays = []
    filesnames = []
    fold = list(os.walk(folder))
    for i in range(1, len(fold)):
        path = fold[i][0] + "/"
        for filename in fold[i][2]:
            img = cv2.imread(path + filename)
            filesnames.append(filename)
            if img is not None:

                height, width, channels = img.shape

                if height > 900 or width > 900:
                    img = cv2.resize(img, (int(height / 2), int(width / 2)))

                images.append(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                grays.append(gray)

    return images, grays, filesnames


def show_data(x, y_, title, xlabel, ylabel, legend, styles):

    y_a, legend_a, styles_a, y_b, legend_b, styles_b = split_methods(y_, legend, styles)

    plot_data(x, y_a, title, xlabel, ylabel, legend_a, styles_a)
    plot_data(x, y_b, title, xlabel, ylabel, legend_b, styles_b)


def plot_data(x, y_, title, xlabel, ylabel, legend, styles):

    # fig, axs = plt.subplots(len(y_), 1)
    plt.title(title)
    if title == "Times Face Recognition":
        plt.ylim(0, 3)

    for i, y in enumerate(y_):
        if title == "Times Face Recognition":
            y.sort(reverse=True)
        plt.plot(x, y, styles[i], linewidth=2)
        plt.legend(legend, loc="right")

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


def split_methods(y, legend, styles):

    tr = ["haar-frontalFace", "haar-profileFace", "LBP-profileFace", "LBP-FrontalFace", "viola-jones"]
    dl = ["HOG", "YOLOv2Face", "FasterRCNNF", "SSDMobileNet", "retinaFace", "RfcnResnet101", "TinyYOLO"]

    y_a = []
    legend_a = []
    styles_a = []

    y_b = []
    legend_b = []
    styles_b = []

    for i, method in enumerate(legend):
        if method in tr:
            y_a.append(y[i])
            legend_a.append(method)
            styles_a.append(styles[i])

        if method in dl:
            y_b.append(y[i])
            legend_b.append(method)
            styles_b.append(styles[i])

    return y_a, legend_a, styles_a, y_b, legend_b, styles_b


def display_results(file_):
    """
    Draw plot with data results 
    
    """

    data, method_names, n = load_result(file_)
    times, n_predic, accuracys = data
    ran = range(n)

    styles = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-']

    show_data(ran, times, "Times Face Recognition", "Index Image", "Time Seconds", method_names, styles)
    show_data(ran, n_predic, "Predicted Faces", "Index Image", "N Predicted Faces", method_names, styles)
    show_data(ran, accuracys, "Accuracy Face Recognition", "Index Image", "Accuracy Method vs Label_Images", method_names, styles)


def load_result(file_):
    """
    load raw data from method analysis  
    metric for accuracy MSE
    
    *see output_result, accuracy_MSE
    
    """

    f = open(file_, "r")
    alpha = 0.00000000001
    times = []
    n_predic = []
    accuracys = []
    method_names = []

    for line in f:
        method_acc = []
        method_predict = []
        method_time = []

        method, n = line.split(" ")
        method_names.append(method)
        n = int(n)

        acc_accumulative = 0
        predict_accumulative = 0

        for _ in range(n):
            time = f.readline().split(" ")
            time = float(time[1])
            method_time.append(time)
            predict_points = f.readline().split(" ")
            labels_points = f.readline().split(" ")
            accs = accuracy_MSE(labels_points, predict_points)
            acc_accumulative += sum(accs) / (len(accs) + alpha)
            predict_accumulative += (len(predict_points) - 2) / 2
            method_acc.append(acc_accumulative)
            method_predict.append(predict_accumulative)

        accuracys.append(method_acc)
        n_predic.append(method_predict)
        times.append(method_time)

    f.close()

    return [times, n_predic, accuracys], method_names, n


def accuracy_MSE(labels_points, predict_points):
    """
    validate the precision of the predictions assigning them a score
    
    """

    accuracys = []

    for l_point in range(1, len(labels_points) - 2, 2):
        acc = 1000000
        for p_point in range(1, len(predict_points) - 2, 2):
            if -1 == labels_points[1]:
                break

            acc_b = math.sqrt((float(labels_points[l_point]) - float(predict_points[p_point])) ** 2 +
                              (float(labels_points[l_point + 1]) - float(predict_points[p_point + 1])) ** 2)

            if acc_b < acc:
                acc = acc_b

            # transform error into accuracy, GAP 20
            if acc > 20:
                acc = 0
            else:
                acc = 1

            accuracys.append(acc)

    # Crop list with the best accuracys in predictions for match in
    # size with predict_points

    if len(accuracys) > len(predict_points):
        accuracys.sort()
        accuracys = accuracys[:len(predict_points)]

    return accuracys


def output_result(method, filesnames, files, labels, predict, times):

    """
    create an output file to persist the scan results
    
    """

    f = open("./results/face_results.txt", "a+")
    f.write(method + " " + str(len(files)))
    f.write("\n")

    for img in filesnames:
        f.write(img + " ")
        if img in files:
            index = filesnames.index(img)
            f.write(str(times[index]))
            f.write(" seconds \n")
            f.write("predict_points ")
            for point in predict[index]:
                f.write(str(point))
                f.write(" ")

            f.write("\n")
            index = files.index(img)
            f.write("labels_points ")
            for point in labels[index]:
                f.write(str(point))
                f.write(" ")

        else:
            f.write(str(-1))
        f.write("\n")
    f.close()


def HOG_():
    # Histogram of Oriented Gradients (HOG)
    predict, times = HOG.predit_HOG(grays)
    output_result("HOG", filesnames, files, labels, predict, times)


def YOLOv2Face_():
    # YOLOv2Face
    predict, times = faceDetectorHof.predit_FaceDetector(images, YOLOv2FaceDetector(min_confidence=1 - threshold))
    output_result("YOLOv2Face", filesnames, files, labels, predict, times)


def FasterRCNNF_():
    # FasterRCNNF
    predict, times = faceDetectorHof.predit_FaceDetector(images, FasterRCNNFaceDetector(min_confidence=1 - threshold))
    output_result("FasterRCNNF", filesnames, files, labels, predict, times)


def SSDMobileNet_():
    # SSDMobileNet

    predict, times = faceDetectorHof.predit_FaceDetector(images, SSDMobileNetV1FaceDetector(min_confidence=1 - threshold))
    output_result("SSDMobileNet", filesnames, files, labels, predict, times)


def Retina_Face_():
    # Retina_Face
    predict, times = retinaFace_.predit_retina_face(images, threshold, 0)
    output_result("retinaFace", filesnames, files, labels, predict, times)


def RfcnResnet101_():
    # RfcnResnet101
    predict, times = faceDetectorHof.predit_FaceDetector(images, RfcnResnet101FaceDetector(min_confidence=1 - threshold))
    output_result("RfcnResnet101", filesnames, files, labels, predict, times)


def TinyYOLO_():
    # TinyYOLO
    predict, times = faceDetectorHof.predit_FaceDetector(images, TinyYOLOFaceDetector(min_confidence=1 - threshold))
    output_result("TinyYOLO", filesnames, files, labels, predict, times)


def Haar_Frontal_Faces_():
    # Haar_Frontal_Faces
    face_cascade = cv2.CascadeClassifier("./weights_models/haar/haarcascade_frontalface_default.xml")
    predict, times = tradicional_methods.predit_haar(grays, face_cascade)
    output_result("haar-frontalFace", filesnames, files, labels, predict, times)


def Haar_Faces_():
    # Haar_Faces
    face_cascade = cv2.CascadeClassifier("./weights_models/haar/haarcascade_profileface.xml")
    predict, times = tradicional_methods.predit_haar(grays, face_cascade)
    output_result("haar-profileFace", filesnames, files, labels, predict, times)


def LBP_Faces_():
    # LBP_Faces
    face_cascade = cv2.CascadeClassifier("./weights_models/LBP/lbpcascade_profileface.xml")
    predict, times = tradicional_methods.predit_haar(grays, face_cascade)
    output_result("LBP-profileFace", filesnames, files, labels, predict, times)


def LBP_Frontal_Faces_():
    # LBP_Frontal_Faces
    face_cascade = cv2.CascadeClassifier("./weights_models/LBP/lbpcascade_frontalface.xml")
    predict, times = tradicional_methods.predit_haar(grays, face_cascade)
    output_result("LBP-FrontalFace", filesnames, files, labels, predict, times)


def Viola_Jones_():
    # Viola-Jones
    face_cascade = cv2.CascadeClassifier("./weights_models/haar/haarcascade_frontalface_alt.xml")
    predict, times = tradicional_methods.predit_haar(grays, face_cascade)
    output_result("viola-jones", filesnames, files, labels, predict, times)


def start_thead(target):
    """
    to prevent the models from getting the resources of
    the graph in tensorflow 1. *, they are launched as threads

    """
    p = multiprocessing.Process(target=target)
    p.start()
    p.join()


def make_analysis():
    """
    
    By default:
        threshold=0.3
        ./dataset/
        ./labels/face_labels.txt
    
    """

    f = open("./results/face_results.txt", "w")
    f.truncate(0)
    f.close()

    start_thead(HOG_)
    start_thead(YOLOv2Face_)
    start_thead(FasterRCNNF_)
    start_thead(SSDMobileNet_)
    start_thead(Retina_Face_)
    start_thead(RfcnResnet101_)
    start_thead(TinyYOLO_)
    start_thead(Haar_Frontal_Faces_)
    start_thead(Haar_Faces_)
    start_thead(LBP_Faces_)
    start_thead(LBP_Frontal_Faces_)
    start_thead(Viola_Jones_)

    print("done")


files, labels = load_labels(path_labels)
images, grays, filesnames = load_images_from_folder(path)

if __name__ == '__main__':
    display_results("./results/face_results.txt")
    # make_analysis()
