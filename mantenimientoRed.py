
import sys

import os
import threading
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import collections
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import calendar, locale

from keras.src.saving import load_model, register_keras_serializable

locale.setlocale(locale.LC_ALL,'es_ES')

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import tensorflow as tf
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

import psycopg2
from psycopg2 import Error

######################
# Variables Globales #
######################
tiempo_entre_busquedas = 5 #En minutos
datos_nuevos = []
datos_por_planta = []
prediccion_por_planta = []
ultimo_id = 0

@register_keras_serializable()
def custom_loss(y_true, y_pred):
    # Clip predictions to avoid log(0) or log(1) which are undefined.
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    # Calculate binary cross-entropy loss
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return loss


def dividir_datos_por_planta(datosIOT):
    datos_por_planta = []

    planta = 1
    datos_por_planta_desorganizados = []
    datos_planta = []
    for datoIOT in datosIOT:
        if planta != datoIOT[1]:
            datos_por_planta_desorganizados.append(datos_planta)
            datos_planta = []
            planta = datoIOT[1]

        datos_planta.append(datoIOT)

    for datos_planta in  datos_por_planta_desorganizados:
        datos_por_planta.append(juntar_datos_planta(datos_planta))

    return datos_por_planta

def juntar_datos_planta(datos_planta):
    datos_organizados = []
    contador = 0
    lecturaCompleta = []
    for datos in datos_planta:
        contador = contador + 1
        planta = datos[1]
        lecturaCompleta.append(datos)
        if (contador == 16 and planta != 5) or (contador == 11 and planta == 5):
            datos_organizados.append(lecturaCompleta)
            lecturaCompleta = []
            contador = 0


    return datos_organizados
def eliminar_datos_inecesarios(datos_IOT_ordenados_por_planta):
    datos_IOT_refinados_por_planta = []
    for planta in datos_IOT_ordenados_por_planta:

        tuplasRefinadas = []
        # Recorremos el array principal
        for datos in planta:
            tupla = []
            # Recorremos cada tupla en el subarray y añadimos el tercer valor a la lista
            for dato in datos:
                tupla.append(dato[3])

            # Convertimos la lista de terceros valores en una tupla
            tuplasRefinadas.append(tuple(tupla[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8)))

        array_convertido = [(float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h), float(i)) for a, b, c, d, e, f, g, h, i in tuplasRefinadas]
        datos_IOT_refinados_por_planta.append(array_convertido)

    return datos_IOT_refinados_por_planta

def preparar_datos_normalizados_red(X):
    # Inicializamos un nuevo array para almacenar todas las tuplas
    valores_normalizados_X = []

    max_X = [-1,-1,-1,-1,-1,-1,-1,-1,-1]

    #Buscar los valores maximos
    for sub_array in X:
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for tupla in sub_array:
            contador = 0
            for elemento in tupla:
                if max_X[contador] < elemento:
                    max_X[contador] = elemento

                contador += 1

    # Normalizar

    # Iteramos sobre cada array interno en X
    for sub_array in X:
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for tupla in sub_array:
            contador = 0
            tupla_normalizada = []
            for elemento in tupla:
                valor_normalizado = elemento / max_X[contador]
                tupla_normalizada.append(valor_normalizado)
                contador += 1

            valores_normalizados_X.append(tuple(tupla_normalizada))


    return np.array(valores_normalizados_X)

def desnormalizar_valores(prediciones):
    # Multiplicar por 100
    array_multiplicado = prediciones * 100

    # Redondear los valores del array multiplicado
    array_redondeado = np.round(array_multiplicado, 1)

    array_formateado = []
    for num in array_redondeado:
        array_formateado.append(num[0])

    return array_formateado

def pred_planta(planta_normalizada):
    y_pred_normalizada = modelo.predict(planta_normalizada)

    y_pred = desnormalizar_valores(y_pred_normalizada)

    return y_pred

def crear_grafica(datos_por_planta,pred_por_planta):
    return ""

def recoger_datos_nuevos():
    global datos_nuevos
    print("Recogiendo datos...")

    try:
        conexion = psycopg2.connect(database='PlantasIA', user='postgres', password="@Andriancito2012@")
        cursor = conexion.cursor()

        comando = f'''SELECT * FROM public."DatosIOT"
    	            where id >= {ultimo_id}
                ORDER BY device_id, date, signal_id ASC '''

        cursor.execute(comando)
        datos_nuevos = cursor.fetchall()
    except Error as e:
        print("Error en la conexion: ", e)

    print("Datos recogidos correctamente")

    return datos_nuevos
    #Conexion con SQL
    #Hacer peticion SQL id>ultimo_id
    #Guardar datos en variable global


def mantenimiento():
    global datos_nuevos

    print("Creando graficas...")

    datos_por_planta = dividir_datos_por_planta(datos_nuevos)

    X_IOT = eliminar_datos_inecesarios(datos_por_planta)

    X_normalizado = preparar_datos_normalizados_red(X_IOT)

    for planta_normalizada in X_normalizado:
        prediccion_por_planta.append(planta_normalizada)

    for i in range(len(datos_por_planta)):
        crear_grafica(datos_por_planta[i], prediccion_por_planta[i])


if __name__ == '__main__':
    modelo = load_model('modelo.keras')
    while True:
        recoger_datos = threading.Thread(target=recoger_datos_nuevos)
        recoger_datos.start()
        recoger_datos.join()
        realizar_mantenimiento = threading.Thread(target=mantenimiento)
        realizar_mantenimiento.start()
        realizar_mantenimiento.join()
        # Espera x minutos antes de la próxima llamada
        time.sleep(tiempo_entre_busquedas * 60)
