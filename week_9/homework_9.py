import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

# image preparation 
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_input(x):
    return x / 255.0


# inference ready
interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

# download image
def predict(url):
    img = download_image(url)
    img = prepare_image(img, (150, 150))
    
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)
    
    interpreter.set_tensor(input_idx, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_idx)

    return float(preds[0,0])
    

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result