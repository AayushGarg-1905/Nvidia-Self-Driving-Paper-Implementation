print('Setting UP')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import torch
from torchvision import transforms

from model import NvidiaModel

sio = socketio.Server()
app = Flask(__name__)
app = socketio.Middleware(sio, app)

maxSpeed = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NvidiaModel().to(device)
model.load_state_dict(torch.load('checkpointsV2/nvidia_model_epoch21.pth', map_location=device))
model.eval()
print("Model loaded")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((66, 200)),
    transforms.ToTensor()
])

def preProcess(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = img[60:-25, :, :]
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img / 127.5 - 1.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])

        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)

        image = preProcess(image)
        with torch.no_grad():
            steering = model(image).item()

        throttle = 1.0 - speed / maxSpeed
        print(f'Steering: {steering:.4f}, Throttle: {throttle:.2f}, Speed: {speed:.2f}')
        sendControl(steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("Connected to simulator")
    sendControl(0.0, 0.0)

def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


