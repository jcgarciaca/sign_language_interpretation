from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, TimeDistributed, Input
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.layers.pooling import GlobalAveragePooling2D

import os
import json
import cv2
import numpy as np

class NetworkModel():
    def __init__(self):
        self.frames = 100
        self.IMG_SIZE = 160
        self.file_path = ''
        
        root_path = '/home/msdc/jcgarciaca/projects/sign_language/cnn_lstm/sign_language_interpretation'
        model_path = os.path.join(root_path, 'exported_models', 'sign-model-02-1.00.h5')
        self.model = load_model(model_path)

        encoding_path = os.path.join(root_path, 'dict', 'encoding.json')
        decoding_path = os.path.join(root_path, 'dict', 'decoding.json')

        self.encoding = {}
        self.decoding = {}
        with open(encoding_path, 'r') as f:
            encoding_tmp = json.load(f)
            for key in encoding_tmp.keys():
                self.encoding[int(key)] = encoding_tmp[key]

        with open(decoding_path, 'r') as f:
            self.decoding = json.load(f)
        
        print('Network model complete!!')

    
    def fit_data(self, data):
        imgs_array = []
        len_imgs = len(data) 
        crop = False
        pad = False
        match = False
        
        difference = 0
        if len_imgs == self.frames:
            match = True
        elif len_imgs > self.frames:
            crop = True
            difference = int((len_imgs - self.frames)/2)
        else:
            pad = True
            difference = int((self.frames - len_imgs)/2)
            
        counter = 0
        for num_img in range(self.frames):
            if pad:
                if num_img < difference or num_img >= (self.frames - difference - 1):
                    # add blank image
                    blank_image = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), np.uint8)
                    imgs_array.append(blank_image)
                    continue
                else:
                    img = data[counter]
                    counter += 1
            elif match:
                img = data[num_img]
            elif crop:
                img = data[num_img + difference]
            
            imgs_array.append(img)
        return imgs_array

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        frames_list = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                img = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                frames_list.append(img)
            else:
                break
                
        X_data = []
        data = []
        for img in frames_list:
            img = cv2.normalize(np.float32(img), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            data.append(img)

        data_fit = self.fit_data(data)
        X_data.append(data_fit)
        X_data = np.array(X_data)
        prediction = self.model.predict(X_data)
        
        if np.max(prediction[0]) > .5:
            key = np.argmax(prediction[0])
            prediction_text = self.encoding[int(key)]
        else:
            prediction_text = 'Not found'
        
        return prediction_text.capitalize()