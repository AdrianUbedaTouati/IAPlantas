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
def obtener_valores(array_de_arrays):
    valores = []
    for subarray in array_de_arrays:
        for i in [3, 5, 6, 7, 8, 9]:  # Solo las posiciones 3, 5, 6, 7, 8, 9
            if 0 <= i < len(subarray):
                valores.append(subarray[i])
    return valores

def main():
    # Array con tuplas que contienen números en formato texto
    array_original = [('2', '3'), ('5', '8'), ('10', '15')]

    # Convertir los números en las tuplas de formato texto a enteros
    array_convertido = [(int(x), int(y)) for x, y in array_original]

    print(array_convertido)

if __name__ == '__main__':
    main()