import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

class FaceDetection(Dataset):
    def __init__(self, imageFolder, annotationFolder, transform=None):
        self.imageFolder = imageFolder
        self.annotationFolder = annotationFolder
        self.transform = transform

        # garante que cada imagem está corretamente "linkada" com seu arquivo de rótulo => essencial para um treino preciso
        self.imagePaths = sorted([os.path.join(imageFolder, imagePath) for imagePath in os.listdir(imageFolder)])
        self.annotationPaths = sorted([os.path.join(annotationFolder, annotationPath) for annotationPath in os.listdir(annotationFolder)])

    def __len__(self):
        return len(self.imagePaths)

    def __get__item(self, idx):
        