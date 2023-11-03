import os
import glob
import shutil
import app.config as config
import cv2 as cv
from PIL import Image
from deepface import DeepFace
from app.utils import remove_representation, check_empty_db

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException, File, UploadFile

import numpy as np

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    if os.path.exists(config.DB_PATH):
        return {
            "message": "Bienvenido!"
        }
    else:
        return {
            "message": f"Error al conectar a {config.DB_PATH}, no hay una base de datos disponible."
        }

@app.get('/img-db-info')
def get_img_db_info(return_img_file: bool | None = True):
    numer_of_images = len(os.listdir(config.DB_PATH))
    pkl_pattern = glob.glob(os.path.join(config.DB_PATH, '*.pkl'))
    pkl_pattern = [file.split('/')[-1] for file in pkl_pattern]

    hidden_pattern = glob.glob(os.path.join(config.DB_PATH, ".*"))
    hidden_pattern = [file.split('/')[-1] for file in hidden_pattern]

    unshow_file = pkl_pattern + hidden_pattern

    if len(pkl_pattern) != 0:
        numer_of_images -= len(unshow_file)

    if return_img_file:
        return {
            "number_of_image": numer_of_images,
            "all_images_file": [file for file in os.listdir(config.DB_PATH) if file not in unshow_file],
        }
    else:
        return {
            "number_of_image": numer_of_images,
        }


@app.get('/show_img/{img_path}')
def show_img(img_path: str | None = None):

    empty = empty = check_empty_db()
    if empty:
        return "Imagen no encontrada en la base de datos"

    if img_path is None:
        return {
            "error": "El usuario deberia de enviar una imagen"
        }

    img_pattern = glob.glob(os.path.join(config.DB_PATH, "*" + img_path + "*"))
    return FileResponse(img_pattern[0])

@app.post('/register')
def face_register(
        img_file: UploadFile | None = File(None, description="Subir imagen"),
        to_gray: bool | None = Query(
            default=True,
            description="Cargar la imagen en escala de grises",),
        img_save_name: str | None = Query(
            default=None,
            description="Nombre de la imagen a guardar",
        ),):
   
    if img_file is None:
        return {
            "message": "Se necesita enviar una imagen!",
        }

    save_img_dir = ''

    # Check save name is correctly
    if img_save_name is not None:

        extension = img_file.filename.split(".")[-1]

        # if img_save_name have extension
        if "." in img_save_name:
            img_save_name_extension = img_save_name.split(".")[-1]
            if extension != img_save_name_extension:
                raise HTTPException(
                    status_code=404, detail='File extension should match')
            save_img_dir = os.path.join(config.DB_PATH, img_save_name)

        # Save name + extension
        else:
            save_img_dir = os.path.join(
                config.DB_PATH, img_save_name + "." + extension)
    # If not save name
    else:
        # Request file name is save name
        if '/' in img_file.filename:
            save_img_dir = os.path.join(
                config.DB_PATH, img_file.filename.split('/')[-1])
        elif "\\" in img_file.filename:
            save_img_dir = os.path.join(
                config.DB_PATH, img_file.filename.split("\\")[-1])
        else:
            save_img_dir = os.path.join(config.DB_PATH, img_file.filename)

    # Raise error if there is duplicate
    if os.path.exists(save_img_dir):
        raise HTTPException(
            status_code=409, detail=f"{save_img_dir} ya esta en la base de datos.")

    # Save image to database
    if (config.RESIZE is False) and (to_gray is False):
        with open(save_img_dir, "wb") as w:
            shutil.copyfileobj(img_file.file, w)

    else:
        try:
            image = Image.open(img_file.file)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            if config.RESIZE:
                image = image.resize(config.SIZE)

            np_image = np.array(image)
            np_image = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)

            if to_gray:
                np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

            cv.imwrite(save_img_dir, np_image)
        except:
            raise HTTPException(
                status_code=500, detail="Sucedio un error al guardar la imagen")
        finally:
            img_file.file.close()
            image.close()

    remove_representation()
    return {
        "message": f"{img_file.filename} fue enviado a {save_img_dir}.",
    }


@app.post("/recognition/")
def face_recognition(
    img_file: UploadFile = File(..., description="Query del archivo imagen"),
    to_gray: bool | None = Query(
        default=True,
        description="Cargar la imagen en escala de grises",),
    return_image_name: bool = Query(
        default=True, description="Retornar el nombre de la imagen"),
):

    empty = check_empty_db()
    if empty:
        return "Base de datos vacia"

    if len(os.listdir(config.DB_PATH)) == 0:
        return {
            "message": "Imagen no enviada a la base de datos."
        }

    if not os.path.exists("query"):
        os.makedirs("query")

    if '/' in img_file.filename:
        query_img_path = os.path.join(
            "query", img_file.filename.split('/')[-1])
    elif "\\" in img_file.filename:
        query_img_path = os.path.join(
            "query", img_file.filename.split("\\")[-1])
    else:
        query_img_path = os.path.join("query", img_file.filename)

    # Convert image to gray (if necessary) then save it
    if to_gray:
        image = Image.open(img_file.file)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        np_image = np.array(image)
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

        cv.imwrite(query_img_path, np_image)
    else:
        with open(query_img_path, "wb") as w:
            shutil.copyfileobj(img_file.file, w)

    df = DeepFace.find(img_path=query_img_path,
                       db_path=config.DB_PATH,
                       model_name=config.MODELS[config.MODEL_ID],
                       distance_metric=config.METRICS[config.METRIC_ID],
                       detector_backend=config.DETECTORS[config.DETECTOR_ID],
                       silent=True, align=True, prog_bar=False, enforce_detection=False)
    
    os.remove(query_img_path)

    if not df.empty:
        path_to_img, metric = df.columns
        ascending = True
        if config.METRIC_ID == 0:
            ascending = False
        df = df.sort_values(by=[metric], ascending=ascending)
        value_img_path = df[path_to_img].iloc[0]

        if return_image_name:
            return_value = value_img_path.split(os.path.sep)[-1]
            return_value = return_value.split(".")[0]
            return {
                "result": return_value,
            }
        else:
            return {
                "result": value_img_path,
            }
    else:
        return {
            "result": "No se encontraron rostros iguales en la base de datos",
        }


@app.put('/change-file-name')
def change_img_name(
    src_path: str = Query(..., description="File image going to be change"),
    img_name: str = Query(..., description="New name")
):
    
    empty = empty = check_empty_db()
    if empty:
        return "Imagen no encontrada en la base de datos"

    src_path = os.path.join(config.DB_PATH, src_path)

    new_path = "/".join(src_path.split("/")[:-1]) + "/" + img_name
    if "." not in img_name:
        extension = src_path.split(".")[1]
        new_path = new_path + "." + extension

    if not os.path.exists(src_path):
        raise HTTPException(
            status_code=404, detail=f'Path to {src_path} is not exist!')

    if os.path.exists(new_path):
        raise HTTPException(
            status_code=409, detail=f"{new_path} already in the database.")

    os.rename(src_path, new_path)

    return {
        "message": f"Already change {src_path} file name to {new_path}"
    }


@app.delete('/del-single-image')
def del_img(img_path: str = Query(..., description="Path to the image need to be deleted")):
    empty = check_empty_db()
    if empty:
        return "No image found in the database"

    img_path = os.path.join(config.DB_PATH, img_path)

    if not os.path.exists(img_path):
        raise HTTPException(
            status_code=404, detail=f'Path to {img_path} is not exist!')

    os.remove(img_path)

    return {
        "message": f"{img_path} has already been deleted!"
    }


@app.delete('/reset-db')
def del_db():
    empty = check_empty_db()
    if empty:
        return "No image found in the database"

    for file in os.listdir(config.DB_PATH):
        os.remove(os.path.join(config.DB_PATH, file))

    if len(os.listdir(config.DB_PATH)) == 0:
        return {
            "message": "All file have been deleted!"
        }
    else:
        raise HTTPException(
            status_code=500, detail="Some thing wrong happened.")
