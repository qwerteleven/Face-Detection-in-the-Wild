Formatos y forma de uso:

main.py:
    el "path" corresponde a una carpeta que contien las carpetas con las imagenes del dataset
    "path_labels", corresponde al archivo de las etiquetas de las posiciones de las caras en las imagenes
    
    
Para crear las etiquetas:
    1. etiquetacion con labelImg -> formato YOLO
    2. eliminar classes.txt creado por labelImg
    3. ejecutar get_point_face_from_YOLO_labels.py, hay que indicar donde estan
       las etiquetas creadas con labelImg
       
       creara un nuevo archivo con la union de todas las etiquetas, guardando
       solo el punto medio en coordenadas relativas donde se encuentra la 
       cara etiquetada
       
       El formato de este archivo sera:  nombre-imagen (x, y)* -> formato YOLO
       
    4. ejecutar center_point_face_labels.py transforma las coordenadas de YOLO a
       coordenadas relativas a la imagen, hay que indicar donde se encuentra el
       archivo anteriormente creado y la ruta del dataset, para poder hacer el
       cambio de coordenadas
       
       El formato sera: nombre-imagen (x, y)* -> coordenadas imagen, redondeo 1 decimal



Para la ejecucion se ejecutar main.py indicando tnato el PATH como el path_labels, con su
correspondiente formato, esto genera un nuevo archivo con el siguiente formato:

    METODO
    Imagen  -> tiempo de ejecucion en segundos
    puntos de caras predichos (x, y)*
    puntos de caras etiquetados (x, y)*   -> si es igual a -1, no hay cara etiquetada 
    
    
Todos los metodos necesitan pesos de configuracion para ser ejecutados
en el caso de RetinaFace se autogestiona, para el resto de casos:

HOG -> "mmod_human_face_detector.dat"

Metodos de cv2 -> Existen dos carpetas "Haar" y "LBP"

Haar contiene los modelos por defecto que trae la instalacion de OpenCV

LBP contine modelos recopilados de modulos opciones de OpenCV




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
