import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import collections
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers

from keras.layers import Dropout

#Wilcoxon Test
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import wilcoxon

#Utilizado para guardar modelos y cargarlos
import joblib

import shap

import cv2

from PIL import Image


def main():
    X = []
    fecha = []
    print("Empezamos")
    for filename in glob.glob(f'../NDVIfotos/*.png'):
        print(filename[18:-7])
        #im = preprocesar_imagen(filename)
        #X.append(image.img_to_array(im))
        # Dejamos de NDVI_2024-02-19_12_32_44 : 2024-02-19_12_32
        fecha.append(filename[18:-7])

if __name__ == '__main__':
    main()