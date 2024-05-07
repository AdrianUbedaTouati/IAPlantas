import sys

import os
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

#Entrenamiento
batch_size = 64
nb_classes = 3
epochs = 100
crossValidationSplit = 2
busquedaMejorTuplaDias = 3

planta_imagen = 1
# numero de tuplas = 5(intervalo entre cada dato)*12( tranformacion a una hora)*24(a un dia)*5(a 5 dias)

# Scaling input image to theses dimensions
img_rows, img_cols = 64, 64

nombreModelo = "modelo1_32x32"

resultadosROC = []

datosIOT = []

datosIOTEntrenamientoImagenes = []

datosOrdenadosPorPlantas = [[], [], [], [], []]  # Lista para cada planta y calidad del aire
humedadIdealOrdenadaPorPlantas = [[], [], [], [], []] # Humedad ideal por cada tupla

input_shape_IOT = (9,) # Datos en entrada en la red

def load_data():
    X = []
    fechas = []

    print("-Procensado imagenes")

    for filename in glob.glob(f'../NDVIfotos/*.png'):
        # Dejamos de NDVI_2024-02-19_12_32_44 : 2024-02-19_12_32
        fechas.append(filename[18:-4])

        im = preprocesar_imagen(filename)
        X.append(image.img_to_array(im))

    input_shape = (img_rows, img_cols, 1)
    return np.array(X), np.array(fechas), input_shape

def preprocesar_imagen(imagen_path):
    # Cargar la imagen utilizando OpenCV
    imagen = cv2.imread(imagen_path)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    alpha = 1.3  # factor de contraste
    beta = -80    # factor de brillo
    imagen_ajustada = cv2.convertScaleAbs(imagen_gris, alpha=alpha, beta=beta)
    # Aplicar umbralización
    #_, imagen_umbralizada = cv2.threshold(imagen_ajustada, 127, 255, cv2.THRESH_BINARY)

    # Aplicar desenfoque
    #imagen_desenfocada = cv2.GaussianBlur(imagen_umbralizada, (5, 5), 0)

    # Aplicar ecualización del histograma
    #imagen_ecualizada = cv2.equalizeHist(imagen_desenfocada)

    # Realizar otras transformaciones si es necesario, como redimensionar, normalizar, etc.
    imagen_final = cv2.resize(imagen_ajustada, (img_rows, img_cols), interpolation=cv2.INTER_AREA)

    X_images_normalized = imagen_final / 255.0

    return X_images_normalized

def plot_symbols(X,y,n=15):
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(n, 3))
    print(X)
    SystemExit(0)
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i]])
        plt.gray()
        ax.set_title('{}-{}'.format(y[index[i]],index[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def obtenerFechaImagenesFormatoSQL(fechasSinProcesar):
    dias = []
    horas = []

    fechasImagenes = []

    for i in range(len(fechasSinProcesar)):
        dias.append(fechasSinProcesar[i][0:-9])
        horas.append(fechasSinProcesar[i][11:])


    for i in range(len(horas)):
        horas[i] = horas[i].replace('_', ':')

    for i in range(len(horas)):
        # Separar la fecha en año, mes y día
        anio = int(dias[i][0:4])
        mes = int(dias[i][5:7])
        dia = int(dias[i][8:10])

        # Separar la hora en hora, minuto y segundo
        hora = int(horas[i][0:2])
        minuto = int(horas[i][3:5])
        segundo = int(horas[i][6:8])

        fechasImagenes.append(datetime(anio, mes, dia, hora, minuto, segundo))

    return fechasImagenes

def obtenerDatosIOT():
    print("-Procensado datos IOT")
    global datosIOT
    try:
        conexion = psycopg2.connect(database = 'PlantasIA', user = 'postgres', password = "@Andriancito2012@")
        cursor = conexion.cursor()
        print("Extrayendo Datos:")
        comando = '''SELECT * FROM public."DatosIOT"
	            where date >= '2024-02-19 12:30:00' and date <= '2024-03-11 12:00:00'
            ORDER BY device_id, date, signal_id ASC '''
        cursor.execute(comando)
        datosIOT = cursor.fetchall()
    except Error as e:
        print("Error en la conexion: ",e)

    return datosIOT

def fecha_mas_cercana(fecha, fechas):
    diferencia_mas_cercana = None
    fecha_mas_cercana = None

    for f in fechas:
        diferencia = abs(fecha - f)
        if diferencia_mas_cercana is None or diferencia < diferencia_mas_cercana:
            diferencia_mas_cercana = diferencia
            fecha_mas_cercana = f

    return fecha_mas_cercana

def tupla_con_fecha_mas_cercana(elemento, plantasOrdenadas):
    diferencia_mas_cercana = None
    tupla_mas_cercana = None

    for dato in plantasOrdenadas:
        fecha = dato[0][4] #lugar donde esta la feha en la tupla
        diferencia = abs(elemento - fecha)
        if diferencia_mas_cercana is None or diferencia < diferencia_mas_cercana:
            diferencia_mas_cercana = diferencia
            tupla_mas_cercana = dato

    return tupla_mas_cercana

def filtrarDatosIOTparaImagenes(fechasImagenes,datosIOT):
    global datosIOTEntrenamientoImagenes
    for i in range(len(fechasImagenes)):
        #print("fecha :",fechasImagenes[i])
        datosIOTEntrenamientoImagenes.append(tupla_con_fecha_mas_cercana(fechasImagenes[i], datosIOT[planta_imagen - 1]))
    return datosIOTEntrenamientoImagenes

#Buscar en la ultima semana de datos
#Cada tupla se registra cada 5 m
# numero de tuplas = 5*12*24*7 = 10 080

#def mejorTupla(fechaActual):
    #Sistema de recompensa

def dividrDatosIOTporPlanta(datosIOT):
    global datosOrdenadosPorPlantas

    contador = 0

    lecturaCompleta = []
    for datoIOT in datosIOT:
        contador = contador + 1
        planta = datoIOT[1]
        lecturaCompleta.append(datoIOT)
        if (contador == 16 and planta != 5) or (contador == 11 and planta == 5):
            datosOrdenadosPorPlantas[planta - 1].append(lecturaCompleta)
            lecturaCompleta = []
            contador = 0

    return datosOrdenadosPorPlantas

def dividrDatosIOTporPlantaImagenes(datosIOT):
    global datosOrdenadosPorPlantas

    contador = 0

    lecturaCompleta = []
    for datoIOT in datosIOT:
        contador = contador + 1
        planta = datoIOT[1]
        lecturaCompleta.append(datoIOT)
        if (contador == 16 and planta != 5) or (contador == 11 and planta == 5):
            datosOrdenadosPorPlantas[planta - 1].append(lecturaCompleta)
            lecturaCompleta = []
            contador = 0

    return datosOrdenadosPorPlantas

def asignarHumedadPerfecta():
    global humedadIdealOrdenadaPorPlantas

    for i in range(len(datosOrdenadosPorPlantas)-1):
        print(f"Planta {i+1} procensando...")
        humedadIdealOrdenadaPorPlantas[i] = obtenerHumedadPerfecta(datosOrdenadosPorPlantas[i])

    return humedadIdealOrdenadaPorPlantas

def asignarHumedadPerfectaImagenes(planta):
    global humedadIdealOrdenadaPorPlantas

    print(f"Planta {planta_imagen-1} procensando...")
    humedadIdealOrdenadaPorPlantas = obtenerHumedadPerfecta(planta)

    return humedadIdealOrdenadaPorPlantas

def obtenerHumedadPerfecta(planta):
    global busquedaMejorTuplaDias
    dias = busquedaMejorTuplaDias

    humedadPerfectaPlanta = []
    humedadMasGrandeDato = []
    fechaDeCadaDato = []

    contador = 0
    for dato in planta:
        contador = contador + 1

        tuplasReferencia = tuplasAnteriores(dato[0][4],dias,planta)
        if (len(tuplasReferencia) == 0):
            tuplasReferencia.append(dato)
            tuplasListas = refinamientoDeTupla(tuplasReferencia)
            mejorTupla = calcular_puntuacion(tuplasListas)

            humedadPerfectaPlanta.append(mejorTupla[6])
        else:
            max_humedad = 0
            for tupla in tuplasReferencia:
                if float(tupla[0][3]) > max_humedad:
                    max_humedad = float(tupla[0][3])

            fechaDeCadaDato.append(dato[0][4])

            humedadMasGrandeDato.append(max_humedad)

            tuplasListas = refinamientoDeTupla(tuplasReferencia)
            mejorTupla = calcular_puntuacion(tuplasListas)

            humedadPerfectaPlanta.append(mejorTupla[6])

    #print(humedadPerfectaPlanta)

    #crearGraficasHumedad(humedadPerfectaPlanta,humedadMasGrandeDato,indicePlanta,dias,fechaDeCadaDato,'Humedad mas grande disponible','Humedad perfecta','Humedad mayor posible frente a la "Humedad perfecta"')

    return humedadPerfectaPlanta

def crearGraficasHumedad(humedadPerfectaPlanta,humedadMasGrandeDato,indicePlanta,dias,fechaDeCadaDato,indice_1,indice_2,titulo):
    plt.plot(humedadPerfectaPlanta, label=indice_1, marker='o', color='blue')  # Puntos de array1
    plt.plot(humedadMasGrandeDato, label=indice_2, marker='o', color='red')  # Puntos de array2

    # Añadir título y etiquetas
    plt.title(titulo)
    plt.xlabel('Indice array')
    plt.ylabel('Humedad')

    #plt.savefig(f"Graficas/Rango_{dias}_dias_comparacion_humedades_planta_{indicePlanta + 1}.png")

    # Mostrar leyenda
    plt.legend()

    # Mostrar el gráfico
    plt.show()

    """
        #Graficas con las fechas
        fig, ax = plt.subplots()

        # Graficar
        ax.plot(fechaDeCadaDato,humedadPerfectaPlanta, label='Array 1', color='blue')  # Puntos de array1
        ax.plot(fechaDeCadaDato,humedadMasGrandeDato, label='Array 2',  color='red')  # Puntos de array2
        ax.set_xlabel('Fechas')
        ax.set_ylabel('Humedad')
        ax.set_title('Humedad en función de las fechas')

        # Configurar el formato automático de fechas en el eje x
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()  # Rotar las fechas para mejor visualización

        plt.tight_layout()  # Ajustar diseño
        plt.show()
    """

def tuplasAnteriores(fecha, dias, tuplas):
    fecha_limite = fecha - timedelta(days=dias)
    tuplas_posteriores = []

    for tupla in tuplas:
        if fecha_limite < tupla[0][4] < fecha:
            tuplas_posteriores.append(tupla)


    return tuplas_posteriores

def refinamientoDeTupla(tuplas):
    tuplasRefinadas = []
    # Recorremos el array principal
    for datos in tuplas:
        tupla = []
        # Recorremos cada tupla en el subarray y añadimos el tercer valor a la lista
        for dato in datos:
            tupla.append(dato[3])

        # Convertimos la lista de terceros valores en una tupla
        tuplasRefinadas.append(tuple(tupla[i] for i in (2, 4, 5, 6, 7, 8, 0)))

    array_convertido = [(float(a), float(b), float(c), float(d), float(e), float(f), float(g)) for a, b, c, d, e, f, g in tuplasRefinadas]

    return array_convertido

def calcular_puntuacion(tuplas):
    # Calcula el valor máximo para cada posición
    valores_maximos = [max(t[i] for t in tuplas) for i in range(6)]

    # Calcula las puntuaciones para cada tupla
    puntuaciones = []
    for tupla in tuplas:
        puntuacion_tupla = 0
        for i in range(6):
            # Calcula la puntuación para cada valor de la tupla
            if tupla[i] == valores_maximos[i]:
                puntuacion_tupla += 1
            else:
                porcentaje = 1 - abs(tupla[i] - valores_maximos[i]) / valores_maximos[i]
                puntuacion_tupla += porcentaje
        # Añade la puntuación de la tupla a la lista de puntuaciones
        puntuaciones.append(puntuacion_tupla)

    # Encuentra la mejor tupla basada en las puntuaciones
    mejor_puntuacion = max(puntuaciones)
    mejor_tupla_index = puntuaciones.index(mejor_puntuacion)
    mejor_tupla = tuplas[mejor_tupla_index]

    return mejor_tupla

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

def eliminar_datos_inecesarios_imagenes(datos_IOT_ordenados_por_planta):
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

        array_convertido = [(float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h), float(i))
                            for a, b, c, d, e, f, g, h, i in tuplasRefinadas]
        datos_IOT_refinados_por_planta.append(array_convertido)

    return datos_IOT_refinados_por_planta


def preparar_datos_red(X,y):
    # Inicializamos un nuevo array para almacenar todas las tuplas
    nuevo_X = []
    nuevo_y = []

    # Iteramos sobre cada array interno
    for sub_array in X:
        # Iteramos sobre cada tupla dentro del array interno y las agregamos al nuevo array
        for tupla in sub_array:
            nuevo_X.append(tupla)

    # Iteramos sobre cada array interno
    for sub_array in y:
        # Iteramos sobre cada tupla dentro del array interno y las agregamos al nuevo array
        for tupla in sub_array:
            nuevo_y.append(tupla)

    return nuevo_X,nuevo_y


def preparar_datos_normalizados_red(X, y):
    # Inicializamos un nuevo array para almacenar todas las tuplas
    nuevo_X = []
    nuevo_y = []

    max_X = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    max_y = 100

    #Buscar los valores maximos
    for sub_array in X:
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for tupla in sub_array:
            contador = 0
            for elemento in tupla:
                if max_X[contador] < elemento:
                    max_X[contador] = elemento

                contador += 1

    """
    # Iteramos sobre cada array interno en y
    for sub_array in y:
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for elemento in sub_array:
            if max_y < elemento:
                max_y = elemento
    """
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

            nuevo_X.append(tuple(tupla_normalizada))


    # Iteramos sobre cada array interno en y
    for sub_array in y:
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for elemento in sub_array:
            normalized_tupla = elemento / max_y
            nuevo_y.append(normalized_tupla)

    return np.array(nuevo_X),np.array(nuevo_y)


def cnn_model_IOT(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(256, activation='relu')(inputs)

    x = layers.Dense(128, activation='relu')(x)

    x = layers.Dense(64,  activation='relu')(x)

    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(1,activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def cnn_model_imagenes_IOT(input_shape_image, input_shape_tuple):

    # Capa de entrada para la imagen
    input_image = layers.Input(shape=input_shape_image)
    # Capa convolucional para procesar la imagen
    conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_image)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    flatten1 = layers.Flatten()(pool2)

    # Capa de entrada para la tupla
    input_tuple = layers.Input(shape=input_shape_tuple)
    flatten2 = layers.Flatten()(input_tuple)

    # Concatenación de las salidas de las capas anteriores
    concatenated = layers.concatenate([flatten1, flatten2])

    dense1 = layers.Dense(128, activation='relu')(concatenated)
    output = layers.Dense(1, activation='sigmoid')(dense1)

    # Modelo final
    model = models.Model(inputs=[input_image, input_tuple], outputs=output)
    return model

@register_keras_serializable()
def custom_loss(y_true, y_pred):
    # Clip predictions to avoid log(0) or log(1) which are undefined.
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    # Calculate binary cross-entropy loss
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return loss

def GuardarValoresROCfichero():
    # Escribimos los resultado en un fichero de texto
    with open(f"datosROC{nombreModelo}.txt", 'w') as file:
        # Escribir los elementos en el archivo, separados por espacios
        file.write(' '.join(map(str, resultadosROC)))

##################################################################################
# Mains program
##################################################################################
def mainImagenes():
    print("Recogiendo datos:")
    X_imagenes, fechas, input_shape = load_data()

    fechasImagenes = obtenerFechaImagenesFormatoSQL(fechas)

    datosIOT = obtenerDatosIOT()

    datos_por_Planta = dividrDatosIOTporPlanta(datosIOT)

    datosFiltrados = filtrarDatosIOTparaImagenes(fechasImagenes, datos_por_Planta)

    print(f"Numero de datos esperados :{len(datosFiltrados)}")
    print(f"Numero de datos devueltos :{len(X_imagenes)}")

    arreglo = []

    arreglo.append(datosFiltrados)

    X_IOT = eliminar_datos_inecesarios_imagenes(arreglo)

    humedades = asignarHumedadPerfectaImagenes(datosFiltrados)

    y = []

    y.append(humedades)
    """
    #Mostrar imagenes
    print(X_IOT.shape, 'train samples')
    print(img_rows, 'x', img_cols, 'image size')
    print(input_shape, 'input_shape')
    print(epochs, 'epochs')
    """

    #print(f"Numero de datos esperados :{len(X_imagenes)}")
    #print(f"Numero de datos devueltos :{len(X_IOT)}")
    #print(f"Numero de datos devueltos :{len(y)}")
    #sys.exit(0)

    X, y = preparar_datos_normalizados_red(X_IOT, y)

    # CV - 10
    kf = KFold(n_splits=crossValidationSplit, shuffle=True, random_state=123)

    #register_keras_serializable.register(custom_loss)

    splitEntrenamiento = 1

    for train_index, test_index in kf.split(X, y, X_imagenes):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_imagenes_train, X_imagenes_test = X_imagenes[train_index], X_imagenes[test_index]

        print(f'x_train {X_train.shape} x_test {X_test.shape}')
        print(f'y_train {y_train.shape} y_test {y_test.shape}')

        model = cnn_model_imagenes_IOT(input_shape,input_shape_IOT)
        print(model.summary())

        model.compile(loss=custom_loss, optimizer='adam', metrics=['mae', 'mse'])

        history = model.fit([X_imagenes_train,X_train], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
        # history = model.fit(train_datagen, steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
        #                    validation_data=(X_test, y_test), verbose=2)

        # Obtener las métricas del historial
        acc = history.history['mae']
        val_acc = history.history['val_mae']
        loss = history.history['mse']
        val_loss = history.history['val_mse']

        # Graficar la precisión
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # Graficar la pérdida
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()

        # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

        # Visualizar datos del split
        loss = model.evaluate([X_imagenes_train,X_test], y_test, batch_size=batch_size)
        y_pred = model.predict([X_imagenes_train,X_test])
        # resultadosROC.append(roc_auc_score(y_test, y_pred[:, 1],multi_class='ovr'))
        print(f"Split numero {splitEntrenamiento}:")

        print('Predictions')
        y_pred_int = y_pred.argmax(axis=1)
        print(collections.Counter(y_pred_int), '\n')

        # Obtener 10 índices aleatorios del conjunto de datos de prueba
        random_indices = np.random.choice(len(X_test), size=10, replace=False)

        # Seleccionar 10 ejemplos aleatorios del conjunto de datos de prueba
        X_sample = X_test[random_indices]
        y_sample_true = y_test[random_indices]
        X_sample_image = X_imagenes_test[random_indices]

        # Hacer predicciones en los ejemplos seleccionados
        y_sample_pred = model.predict([X_sample_image,X_sample])

        # Mostrar los resultados
        for i in range(10):
            print("Ejemplo", i + 1)
            print("Valor real:", y_sample_true[i])
            print("Valor predicho:", y_sample_pred[i])
            print("-------------------------")

    model.save('modelo_imagenes.keras')

def main():
    print("Recogiendo datos:")
    datosIOT = obtenerDatosIOT()

    datos_por_Planta = dividrDatosIOTporPlanta(datosIOT)

    X_IOT = eliminar_datos_inecesarios(datos_por_Planta)

    y = asignarHumedadPerfecta()

    #Eliminamos el sensor de C02
    del X_IOT[4]
    del y[4]

    X,y=preparar_datos_normalizados_red(X_IOT,y)

    #CV - 10
    kf = KFold(n_splits=crossValidationSplit, shuffle=True, random_state=123)

    splitEntrenamiento = 1

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f'x_train {X_train.shape} x_test {X_test.shape}')
        print(f'y_train {y_train.shape} y_test {y_test.shape}')

        model = cnn_model_IOT(input_shape_IOT)
        print(model.summary())

        model.compile(loss=custom_loss, optimizer='adam', metrics=['mae', 'mse'])

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
        #history = model.fit(train_datagen, steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
        #                    validation_data=(X_test, y_test), verbose=2)

        # Obtener las métricas del historial
        acc = history.history['mae']
        val_acc = history.history['val_mae']
        loss = history.history['mse']
        val_loss = history.history['val_mse']

        # Graficar la precisión
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # Graficar la pérdida
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()

        #shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

        #Visualizar datos del split
        loss = model.evaluate(X_test, y_test, batch_size=batch_size)
        y_pred = model.predict(X_test)
        #resultadosROC.append(roc_auc_score(y_test, y_pred[:, 1],multi_class='ovr'))
        print(f"Split numero {splitEntrenamiento}:")

        print('Predictions')
        y_pred_int = y_pred.argmax(axis=1)
        print(collections.Counter(y_pred_int), '\n')

        # Obtener 10 índices aleatorios del conjunto de datos de prueba
        random_indices = np.random.choice(len(X_test), size=10, replace=False)

        # Seleccionar 10 ejemplos aleatorios del conjunto de datos de prueba
        X_sample = X_test[random_indices]
        y_sample_true = y_test[random_indices]

        # Hacer predicciones en los ejemplos seleccionados
        y_sample_pred = model.predict(X_sample)

        # Mostrar los resultados
        for i in range(10):
            print("Ejemplo", i + 1)
            print("Valor real:", y_sample_true[i])
            print("Valor predicho:", y_sample_pred[i])
            print("-------------------------")

    model.save('modelo.keras')


def desnormalizar_valores(prediciones):
    # Multiplicar por 100
    array_multiplicado = prediciones * 100

    # Redondear los valores del array multiplicado
    array_redondeado = np.round(array_multiplicado, 1)

    array_formateado = []
    for num in array_redondeado:
        array_formateado.append(num[0])

    return array_formateado


def main_probar_modelo():
    print("Recogiendo datos:")
    datosIOT = obtenerDatosIOT()

    datos_por_Planta = dividrDatosIOTporPlanta(datosIOT)

    X_IOT = eliminar_datos_inecesarios(datos_por_Planta)

    y = asignarHumedadPerfecta()

    X_planta_objetivo = []
    y_planta_objetivo = []

    X_planta_objetivo.append(X_IOT[1])
    y_planta_objetivo.append(y[1])

    X, y = preparar_datos_normalizados_red(X_planta_objetivo, y_planta_objetivo)

    modelo_sin_imagenes = load_model('modelo_sin_imagenes.keras')

    print("Modelo sin imagenes: ")

    y_pred = modelo_sin_imagenes.predict(X)

    print('Predictions')
    y_pred_int = y_pred.argmax(axis=1)
    print(collections.Counter(y_pred_int), '\n')

    y_pred_desnormalizado = desnormalizar_valores(y_pred)

    y_desnormalizado = y * 100

    crearGraficasHumedad(y_pred_desnormalizado, y_desnormalizado, 0, 3, 0,'Predicion realizada', 'Humedad perfecta', 'Predicion realizadas por el modelo frente a la "Humedad perfecta"')

    """
    modelo_imagenes = load_model('modelo_imagenes.keras')

    print("Modelo con imagenes: ")

    y_pred = modelo_imagenes.predict([X)

    print('Predictions')
    y_pred_int = y_pred.argmax(axis=1)
    print(collections.Counter(y_pred_int), '\n')

    y_pred_desnormalizado = desnormalizar_valores(y_pred)

    y_desnormalizado = y * 100

    crearGraficasHumedad(y_pred_desnormalizado, y_desnormalizado, 0, 3, 0, 'Predicion realizada', 'Humedad perfecta',
                         'Predicion realizadas por el modelo frente a la "Humedad perfecta"')
    """

if __name__ == '__main__':
    probar_modelo = False

    entrenar_con_imagen = False

    if probar_modelo:
        main_probar_modelo()
    else:
        if entrenar_con_imagen:
            mainImagenes()
        else:
            main()
