import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import collections
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import calendar, locale

locale.setlocale(locale.LC_ALL,'es_ES')

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
epochs = 50
crossValidationSplit = 10
busquedaMejorTuplaDias = 1000
# numero de tuplas = 5(intervalo entre cada dato)*12( tranformacion a una hora)*24(a un dia)*5(a 5 dias)

# Scaling input image to theses dimensions
img_rows, img_cols = 64, 64

nombreModelo = "modelo1_32x32"

resultadosROC = []

datosIOT = []

datosIOTEntrenamientoImagenes = []

datosOrdenadosPorPlantas = [[], [], [], [], []]  # Lista para cada planta y calidad del aire
humedadIdealOrdenadaPorPlantas = [[], [], [], [], []] # Humedad ideal por cada tupla

def load_data():
    X = []
    fechas = []

    print("Recogiendo datos")
    for filename in glob.glob(f'../NDVIfotos/*.png'):
        # Dejamos de NDVI_2024-02-19_12_32_44 : 2024-02-19_12_32
        fechas.append(filename[18:-4])

        #im = preprocesar_imagen(filename)
        #X.append(image.img_to_array(im))

    input_shape = (img_rows, img_cols, 1)
    return np.array(X), np.array(fechas), input_shape

def preprocesar_imagen(imagen_path):
    # Cargar la imagen utilizando OpenCV
    imagen = cv2.imread(imagen_path)

    imagenRGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    imagenNormalizada = imagenRGB.astype('float32') / 255.0

    # Realizar otras transformaciones si es necesario, como redimensionar, normalizar, etc.
    imagen_final = cv2.resize(imagenNormalizada, (img_rows, img_cols))

    return imagen_final

def plot_symbols(X,y,n=15):
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(n, 3))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i]])
        #plt.gray()
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

def fecha_mas_cercana(fecha, fechas):
    diferencia_mas_cercana = None
    fecha_mas_cercana = None

    for f in fechas:
        diferencia = abs(fecha - f)
        if diferencia_mas_cercana is None or diferencia < diferencia_mas_cercana:
            diferencia_mas_cercana = diferencia
            fecha_mas_cercana = f

    return fecha_mas_cercana

def tupla_con_fecha_mas_cercana(elemento, array_de_tuplas):
    diferencia_mas_cercana = None
    tupla_mas_cercana = None

    for tupla in array_de_tuplas:
        fecha = tupla[4] #lugar donde esta la feha en la tupla
        diferencia = abs(elemento - fecha)
        if diferencia_mas_cercana is None or diferencia < diferencia_mas_cercana:
            diferencia_mas_cercana = diferencia
            tupla_mas_cercana = tupla

    return tupla_mas_cercana

def filtrarDatosIOTparaImagenes(fechasImagenes):
    global datosIOTEntrenamientoImagenes
    for i in range(len(fechasImagenes)):
        #print("fecha :",fechasImagenes[i])
        datosIOTEntrenamientoImagenes.append(tupla_con_fecha_mas_cercana(fechasImagenes[i],datosIOT))

#Buscar en la ultima semana de datos
#Cada tupla se registra cada 5 m
# numero de tuplas = 5*12*24*7 = 10 080

#def mejorTupla(fechaActual):
    #Sistema de recompensa

def dividrDatosIOTporPlanta():
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

def asignarHumedadPerfecta():
    global humedadIdealOrdenadaPorPlantas

    for i in range(len(datosOrdenadosPorPlantas)-1):
        print(f"Planta {i+1} procensado")
        humedadIdealOrdenadaPorPlantas[i] = obtenerHumedadPerfecta(datosOrdenadosPorPlantas[i],i)

def obtenerHumedadPerfecta(planta,indicePlanta):
    global busquedaMejorTuplaDias
    dias = busquedaMejorTuplaDias

    humedadPerfectaPlanta = []
    humedadMasGrandeDato = []
    fechaDeCadaDato = []

    contador = 0
    for dato in planta:
        contador = contador + 1

        tuplasReferencia = tuplasAnteriores(dato[0][4],dias,planta)
        #print("Tuplas refenriacia longitud: "+str(len(tuplasReferencia))+ " contenido dato: " +str(dato[0])+ " dias: "+str(dias))
        if (len(tuplasReferencia) == 0):
            mejorTupla = dato
        else:
            max_humedad = 0
            for tupla in tuplasReferencia:
                if float(tupla[0][3]) > max_humedad:
                    max_humedad = float(tupla[0][3])

            fechaDeCadaDato.append(dato[0][4])

            #print(max_humedad)
            humedadMasGrandeDato.append(max_humedad)
            tuplasListas = refinamientoDeTupla(tuplasReferencia)
            #print(dato)
            #print(tuplasListas)
            mejorTupla = calcular_puntuacion(tuplasListas)
            humedadPerfectaPlanta.append(mejorTupla[6])

    #print(len(humedadPerfectaPlanta))
    #print(humedadPerfectaPlanta)

    #print(len(humedadMasGrandeDato))
    #print(humedadMasGrandeDato)

    #print(fechaDeCadaDato)

    # Crear el gráfico
    plt.plot(humedadPerfectaPlanta, label='Humedad perfecta', marker='o', color='blue')  # Puntos de array1
    plt.plot(humedadMasGrandeDato, label='Humedad mas grande disponible', marker='o', color='red')  # Puntos de array2

    # Añadir título y etiquetas
    plt.title('Humedad mayor posible frente a la "Humedad perfecta"')
    plt.xlabel('Indice array')
    plt.ylabel('Humedad')

    plt.savefig(f"Graficas/Rango_{dias}_dias_comparacion_humedades_planta_{indicePlanta+1}.png")

    # Mostrar leyenda
    plt.legend()

    # Mostrar el gráfico
    plt.show()

    """
        #Graficas con las fechas
        fig, ax = plt.subplots()

        # Graficar
        ax.plot(fechaDeCadaDato,humedadPerfectaPlanta, label='Array 1', marker='o', color='blue')  # Puntos de array1
        ax.plot(fechaDeCadaDato,humedadMasGrandeDato, label='Array 2', marker='o', color='red')  # Puntos de array2
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

    return humedadPerfectaPlanta


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

def cnn_model(input_shape, nb_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(nb_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def GuardarValoresROCfichero():
    # Escribimos los resultado en un fichero de texto
    with open(f"datosROC{nombreModelo}.txt", 'w') as file:
        # Escribir los elementos en el archivo, separados por espacios
        file.write(' '.join(map(str, resultadosROC)))

##################################################################################
# Main program
def main():
    X, y, input_shape = load_data()

    print(X.shape, 'train samples')
    print(img_rows,'x', img_cols, 'image size')
    print(input_shape,'input_shape')
    print(epochs,'epochs')

    #plot_symbols(X, y)
    collections.Counter(y)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    fechasImagenes = obtenerFechaImagenesFormatoSQL(y)

    obtenerDatosIOT()

    filtrarDatosIOTparaImagenes(fechasImagenes)

    dividrDatosIOTporPlanta()

    asignarHumedadPerfecta()

    """
    # CNN layer need an additional chanel to colors (32 x 32 x 1)
    print('N samples, witdh, height, channels',X.shape)

    #El data set contiene 1583+4273=5856 imagenes
    kf = StratifiedKFold(n_splits=crossValidationSplit, shuffle=True, random_state=123)

    splitEntrenamiento = 1

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)

        print("Número de imágenes generadas:", len(train_datagen) * batch_size)

        print(f'x_train {X_train.shape} x_test {X_test.shape}')
        print(f'y_train {y_train.shape} y_test {y_test.shape}')

        model = cnn_model(input_shape, nb_classes)
        print(model.summary())

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

        #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
        history = model.fit(train_datagen, steps_per_epoch=len(X_train) // batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=2)

        # Obtener las métricas del historial
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

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

        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

        #Visualizar datos del split
        loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
        y_pred = model.predict(X_test)
        #resultadosROC.append(roc_auc_score(y_test, y_pred[:, 1],multi_class='ovr'))
        resultadosROC.append(roc_auc_score(y_test, y_pred, multi_class='ovr'))
        print(f"Split numero {splitEntrenamiento}:")
        print(f'loss: {loss:.2f} acc: {acc:.2f}')
        #print(f'AUC {roc_auc_score(y_test, y_pred[:, 1], ):.4f}')
        print(f'AUC {resultadosROC[splitEntrenamiento-1]:.4f}')

        print('Predictions')
        y_pred_int = y_pred.argmax(axis=1)
        print(collections.Counter(y_pred_int), '\n')

        print('Metrics')
        print(metrics.classification_report(y_test, y_pred_int, target_names=['Normal', 'Pneumonia_bacteriana', 'Pneumonia_viral']))

        print('Confusion matrix')
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred_int),
                                       display_labels=['NORMAL', 'PNEUMONIA_BACTERIANA','PNEUMONIA_VIRAL']).plot()
        plt.show()

        # Selecciona 10 imágenes al azar de X_test
        indices = np.random.choice(np.arange(len(X_test)), size=10, replace=False)
        X_explain = X_test[indices]

        # Usa solo un subset de X_train
        X_train_subset = X_train[:1000]
        explainer = shap.DeepExplainer(model, X_train_subset)

        # Calcula los valores SHAP
        shap_values = explainer.shap_values(X_explain)

        # Visualiza los valores SHAP
        shap.image_plot(shap_values, -X_explain)

        splitEntrenamiento += 1

    GuardarValoresROCfichero()

    print("Fin de entrenamiento")
    """

if __name__ == '__main__':
    main()
