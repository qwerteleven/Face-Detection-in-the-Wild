import cv2
import os


def generate_labels_file(folder_images, folder_labels):

    detections, labels_img = load_labels_data(folder_labels)
    img_shapes, img_names = load_img_data(folder_images)
    labels_point_center(img_names, img_shapes, labels_img, detections)


def labels_point_center(img_names, img_shapes, labels_img, detections):

    """
    Create a file with the center points of detections
    
    """
    
    center_point = []

    for img_name in img_names:
        points = [img_name]
        if img_name in labels_img:
            index = labels_img.index(img_name)
            points = detections[index]
            width, height = img_shapes[index]
            
            for i in range(0, len(points), 2):
                x = width * points[i]
                y = height * points[i + 1]
                points.append(str(round(x, 2)))
                points.append(str(round(y, 2)))
        else:
            points.append("-1") # not found label for img, or not detection
                
        center_point.append(points)
    
    write_file("face_labels.txt", center_point)
    
    
def write_file(file_, instances):

    """
    Write a file, truncating it first 
    
    one line for instance
    
    """
    f = open(file_, 'x')
    f.truncate(0) 
                
    for values in instances:
        for value in values:
            f.write(value)
            f.write(" ")
        f.write("\n")
    f.close()       
    
                 
def load_img_data(folder):

    """
    Load shapes of images
    
    """
    shapes = []
    img_names = []
    
    fold = list(os.walk(folder))
    for i in range(1, len(fold)):
        path = fold[i][0] + "/"
        for filename in fold[i][2]:
            img = cv2.imread(path + filename)
            if img is not None:
                height, width, channels = img.shape
                shapes.append([height, width])
                img_names.append(filename.split(".")[0])
            
    return shapes, img_names    
            

def load_labels_data(folder):

    """
    Load all labels information of YOLO
    
    """
    
    labels = []
    img_names = []
    p = []
    
    for entry in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, entry)):
            f = open(folder + entry, "r")
            lines = f.readlines()
            img_names.append(entry.split(".")[0])
            
            for line in lines:
                value = line.split(" ")
                
                for i in range(2):
                    labels.append(value[i + 1])
                    
            p.append(labels)
            labels = []
            f.close()
            
    return p, img_names
