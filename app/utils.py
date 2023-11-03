import os
import glob
import app.config as config
from PIL import Image

def remove_representation():
    representations_path = glob.glob(os.path.join(config.DB_PATH, "representations_*.pkl"))
    if len(representations_path) != 0:
        for representation in representations_path:
            if os.path.exists(representation):
                os.remove(representation)

def check_empty_db():
    if len(os.listdir(config.DB_PATH)) == 0:
        return True
    return False

def show_img(input_path:str):

    if not isinstance(input_path, str):
        raise TypeError("Only string is accepted, expect an input path as string.")
    
    if not os.path.exists(input_path):
        raise ValueError('Path to the image is not available.')

    try:
        image = Image.open(input_path)
        image.show()
    except:
        print("Error when reading image, check input_path.")