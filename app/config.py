METRIC_ID = 2
MODEL_ID = 6
DETECTOR_ID = 4

DB_PATH = "app/data"  
RESIZE = False
SIZE = (300, 300)

METRICS = [
  "cosine",   
  "euclidean",  
  "euclidean_l2"  
  ]
MODELS = [
  "VGG-Face",  
  "Facenet",  
  "Facenet512",   
  "OpenFace",   
  "DeepFace",  
  "DeepID",   
  "ArcFace",  
  "Dlib",   
  "SFace",  
]
DETECTORS = [
  'opencv',  
  'ssd',  
  'dlib',   
  'mtcnn',  
  'retinaface', 
  'mediapipe' 
]