from fastapi import FastAPI, UploadFile, File, HTTPException
from uuid import uuid4
import cv2
import boto3
import numpy as np
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
import requests
import io
import sys
sys.path.append('./app')
from utils import normalized_coords


app = FastAPI()

model_det = YOLO('./weights/weight.pt') ##определение модели детекции

class DetectionInput(BaseModel):
    url: str


@app.post("/")
async def detect_borders(input: DetectionInput):
    ##считывание изображения
    url = input.url
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img = np.array(img)
        else:
            HTTPException(status_code=404, detail="Failed to download image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    ##получение результата детекции
    result = model_det(img, verbose=False)[0]
    boxes = result.obb.xyxyxyxy
    ##реализация non max suppression
    n_max_conf = [x for x in result.obb.conf.numpy()]
    if len(n_max_conf) > 0:
        n_max_conf = max(n_max_conf)
    else:
        n_max_conf = 0
    nms = list(map(lambda x: x == n_max_conf, result.obb.conf.numpy()))
    boxes = np.array(boxes[nms], np.int32)
    ##проверка наличия задетектированного бокса
    if (len(boxes) > 0):
        a, b, c, d = normalized_coords(boxes[0])
    else:
        a, b, c, d = [0, 0], [0, 112], [512, 0], [512, 112]
    pts1 = np.float32([a, b,
                       c, d])
    pts2 = np.float32([[0, 0], [0, 112],
                       [512, 0], [512, 112]])
    ##подсчет матрицы перспективы и последующее изменение перспективы номера при помощи координат углов
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (512, 112))

    ##отправка преобразованного изображения обратно на сервер
    s3 = boto3.client("s3",
                       aws_access_key_id='BLZVPJ5JNCHVJZPR6SU3',
                       aws_secret_access_key='1dwymiu3T0y95gQSm3ivnQHnqHqatJPAyZIyqA4p',
                       region_name='ru-1',
                       endpoint_url='https://s3.timeweb.cloud')
    bucket_name = "516d5635-4ecebcb3-728f-458d-a2e8-786f8949b0d2"
    uid = str(uuid4())
    object_name = uid + "detectedphoto.png"
    data_serial = cv2.imencode('.png', result)[1].tobytes()
    file_obj = io.BytesIO(data_serial)
    s3.upload_fileobj(file_obj, bucket_name, object_name)

    object_link = 'https://s3.timeweb.cloud/' + bucket_name + '/' + object_name

    return {"result_link": object_link}