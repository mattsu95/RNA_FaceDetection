import tensorflow as tf
import numpy as np
import cv2

# from ultralytics import YOLO

# import torch


class AgeDetection:

    def __init__(self):
        print("Carregando modelo...")
        self.age_model = tf.keras.models.load_model("models/age_model_acc_0.762.h5")
        print("Modelo carregado com sucesso!")

        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

        self.classes_idade = ["1-2", "3-9", "10-20", "21-27", "28-45", "46-65", "66-116"]

    def detect_by_img(self, img_path):
        if img_path.size == 0:
            return ""

        # preprocessamento obrigatório
        face_gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (200, 200))
        face_input = np.expand_dims(face_resized, axis=[0, -1])

        # Faz a predição
        age_predictions = self.age_model.predict(face_input, verbose=0)
        class_index = np.argmax(age_predictions)

        return self.classes_idade[class_index]