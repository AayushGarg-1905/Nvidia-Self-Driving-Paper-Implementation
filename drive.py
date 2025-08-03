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
model.load_state_dict(torch.load('checkpoints/nvidia_model_epoch10.pth', map_location=device))
model.eval()
print("Model loaded")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((66, 200)),
    transforms.ToTensor()
])

def preProcess(img):
    img = img[60:-25, :, :]
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = img / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / std

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















































































































# # drive.py

# import argparse
# import base64
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import torch
# import cv2

# from flask import Flask


# from model import NvidiaModel
# from torchvision import transforms


# # Setup Flask app
# app = Flask(__name__)
# socketio = SocketIO(app)

# # Image preprocessing
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((66, 200)),
#     transforms.ToTensor()
# ])

# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NvidiaModel().to(device)
# model.load_state_dict(torch.load('checkpoints/nvidia_model_epoch10.pth', map_location=device))
# model.eval()
# print('--------------------------model loaded')
# @socketio.on('telemetry')
# def telemetry(data):
#     print("---------------------------------------Received telemetry") 
#     if data:
#         # Decode image
#         img_str = data["image"]
#         image = Image.open(BytesIO(base64.b64decode(img_str)))
#         image = np.asarray(image)

#         # Preprocess
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         image = preprocess(image).unsqueeze(0).to(device)

#         # Predict
#         with torch.no_grad():
#             steering_angle = model(image).item()

#         print(f"Predicted angle: {steering_angle:.4f}")

#         # Send back to simulator
#         send_control(steering_angle, 5.0)

# @socketio.on('connect')
# def connect():
#     print("----------------------------------------------Connected to Udacity Simulator.")
#     send_control(0.0, 5.0)

# def send_control(steering_angle, throttle):
#     print('------------------send control')
#     socketio.emit(
#         "steer",
#         data={
#             'steering_angle': str(steering_angle),
#             'throttle': str(throttle)
#         }
#     )

# if __name__ == '__main__':
#     print('--------------------------module started')
#     socketio.run(app, host='0.0.0.0', port=4567, debug=True, use_reloader=False)
