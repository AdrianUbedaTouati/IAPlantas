
import sys
import os
import threading
import time

import numpy

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

#Graficas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Javascript
import plotly.io as pio
import webbrowser

######################
# Variables Globales #
######################
tiempo_entre_busquedas = 0.5 #En minutos
datos_nuevos = []
datos_por_planta = []
prediccion_por_planta = []
ultimo_id = 0

######################
# Variables Graficos #
######################
pagina_abierta = False
alterta_diferencia_humedad_roja = 50
alterta_diferencia_humedad_amarilla = 25
num_columnas_pagina = 2
num_lineas_pagina = 0 #automatico

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

        if datoIOT[0] == 1610213:
            print("error") #pasado

        datos_planta.append(datoIOT)


    #ultima planta
    datos_por_planta_desorganizados.append(datos_planta)

    # borramos el sesonr 4
    del datos_por_planta_desorganizados[4]

    for datos_planta in datos_por_planta_desorganizados:
        datos_por_planta.append(juntar_datos_planta(datos_planta))

    dato_problematico = False

    """
    for datos_planta in datos_por_planta:
        datos_planta[:] = [tuplas for tuplas in datos_planta if
                           all(verificar_rango(int(dato[2] % 16), float(dato[3])) for dato in tuplas)]
    
    for datos_planta in datos_por_planta:
        datos_planta_filtrados = []
        for tuplas in datos_planta:
            for dato in tuplas:
                if not(verificar_rango(int(dato[2] % 16), float(dato[3]) )):
                    dato_problematico = True
                    break
            if dato_problematico:
                del tuplas
                dato_problematico = False
    """

    return datos_por_planta

def verificar_rango(sensor,valor):
    resultado = False

    rango_humedad = [0,100]
    rango_temperatura = [-20, 80]
    rango_conductividad = [0, 3500]
    rango_ph = [0, 14]

    rango_nitrogeno = [0, 800]
    rango_fosforo = [0, 1800]
    rango_potasio = [0, 1800]
    rango_salinidad = [0, 2100]
    rango_tds = [0, 1900]

    if sensor == 1:
        if valor >= rango_humedad[0] and valor <= rango_humedad[1]:
            resultado = True

    elif sensor == 2:
        if valor >= rango_temperatura[0] and valor <= rango_temperatura[1]:
            resultado = True

    elif sensor == 3:
        if valor >= rango_conductividad[0] and valor <= rango_conductividad[1]:
            resultado = True

    elif sensor == 4:
        if valor >= rango_ph[0] and valor <= rango_ph[1]:
            resultado = True

    elif sensor == 5:
        if valor >= rango_nitrogeno[0] and valor <= rango_nitrogeno[1]:
            resultado = True

    elif sensor == 6:
        if valor >= rango_fosforo[0] and valor <= rango_fosforo[1]:
            resultado = True

    elif sensor == 7:
        if valor >= rango_potasio[0] and valor <= rango_potasio[1]:
            resultado = True

    elif sensor == 8:
        if valor >= rango_salinidad[0] and valor <= rango_salinidad[1]:
            resultado = True

    elif sensor == 9:
        if valor >= rango_tds[0] and valor <= rango_tds[1]:
            resultado = True

    else:
        resultado = True

    return resultado


def juntar_datos_planta(datos_planta):
    datos_organizados = []
    contador = 0
    anterior_sensor = -1;
    lecturaCompleta = []
    for datos in datos_planta:
        # Hay veces que viene el mismo dato de 2 dispositivos distintos veces seguidas super raro, 4 dias depurando
        if datos[2] != anterior_sensor:
            #if datos[0] == 1610213:
            #    print("error")
            contador = contador + 1
            anterior_sensor = datos[2]
            lecturaCompleta.append(datos)
            if contador == 16:
                lectura_ordenada = sorted(lecturaCompleta, key=lambda x: x[2])
                datos_organizados.append(lectura_ordenada)
                lecturaCompleta = []
                contador = 0

    return datos_organizados

"""
def juntar_datos_planta(datos_planta):
    numero_de_datos_tupla = 16
    principio_tupla = 1
    final_tupla = 16

    datos_planta_por_tuplas = []
    planta = -1

    lectura_tupla = []
    for dato in datos_planta:
        if planta != dato[1]:
            if(principio_tupla > 1):
                print("error")

            planta = dato[1]
            final_tupla = numero_de_datos_tupla * planta
            principio_tupla = final_tupla - 15

            lectura_tupla = []

        if principio_tupla == final_tupla:
            lectura_tupla.append(dato)
            datos_planta_por_tuplas.append(lectura_tupla)
            principio_tupla = final_tupla - 15
            lectura_tupla = []
        else:
            lectura_tupla.append(dato)

            principio_tupla = principio_tupla + 1

    return datos_planta_por_tuplas
"""

def eliminar_datos_inecesarios(datos_IOT_ordenados_por_planta):
    datos_IOT_refinados_por_planta = []
    id = -1
    primero = True
    for planta in datos_IOT_ordenados_por_planta:

        tuplasRefinadas = []
        # Recorremos el array principal
        for tupla_datos in planta:

            tupla = []
            # Recorremos cada tupla en el subarray y añadimos el tercer valor a la lista
            for dato in tupla_datos:
                if primero:
                    id = dato[0]
                    primero = False

                tupla.append(dato[3])

            tupla.append(id)

            if tupla[16] == 1610213:
                print(tupla)

            primero = True

            # Convertimos la lista de terceros valores en una tupla
            tupla_refinada = tuple(tupla[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 16))
            tuplasRefinadas.append(tupla_refinada)

        array_convertido = [(float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h), float(i), int(j)) for a, b, c, d, e, f, g, h, i ,j in tuplasRefinadas]
        datos_IOT_refinados_por_planta.append(array_convertido)

        #for tuplaa in array_convertido:
        #    if tuplaa[9] == 1610213:
        #        print(tuplaa[9])


    return datos_IOT_refinados_por_planta

def preparar_datos_normalizados_red(X):
    # Inicializamos un nuevo array para almacenar todas las tuplas
    datos_por_planta_normalizados = []

    max_X = [100,-1,-1,-1,-1,-1,-1,-1,-1] #ultimo es el id luego lo borramos

    max_X_por_planta = []

    max_X_anterior = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    max_X_por_planta_anterior = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    dato_problematico = False

    #Buscar los valores maximos
    for planta in X:
        max_X_planta = [-1,-1,-1,-1,-1,-1,-1,-1,-1] #ultimo es el id luego lo borramos
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for tupla in planta:
            contador = 0
            for elemento in tupla:
                # no tener en cuenta el id
                if not(verificar_rango(contador + 1,elemento)):
                    dato_problematico = True
                    break
                if contador == 9:
                    break

                if max_X_planta[contador] < elemento:
                    max_X_planta[contador] = elemento

                if max_X[contador] < elemento:
                    max_X[contador] = elemento

                contador += 1

            if dato_problematico:
                dato_problematico = False
                max_X = max_X_anterior
                max_X_planta = max_X_por_planta_anterior
            else:
                max_X_anterior = max_X
                max_X_por_planta_anterior = max_X_planta

        max_X_por_planta.append(max_X_planta)

    print("General")
    print(max_X)
    print("Por planta")
    for max_planta in max_X_por_planta:
        print(max_planta)
    # Normalizar

    # Iteramos sobre cada array interno en X
    for planta in X:
        valores_normalizados_planta = []
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for tupla in planta:
            contador = 0
            tupla_normalizada = []
            for elemento in tupla:
                # no tener en cuenta el id
                if contador == 9:
                    break
                valor_normalizado = elemento / max_X[contador]
                tupla_normalizada.append(valor_normalizado)
                contador += 1

            valores_normalizados_planta.append(tuple(tupla_normalizada))

        datos_por_planta_normalizados.append(np.array(valores_normalizados_planta))


    return datos_por_planta_normalizados

def desnormalizar_valores(prediciones):
    # Multiplicar por 100
    array_multiplicado = prediciones * 100

    # Redondear los valores del array multiplicado
    array_redondeado = np.round(array_multiplicado, 1)

    array_formateado = []
    for num in array_redondeado:
        array_formateado.append(num[0])

    return array_formateado

def prediccion_plantas(planta_normalizada):
    y_pred_normalizada = modelo.predict(planta_normalizada)

    y_pred = desnormalizar_valores(y_pred_normalizada)

    return y_pred

def obtener_humedad(plantas_normalizadas):
    humedad_por_plantas = []

    for datos_planta in plantas_normalizadas:
        planta = []
        for tupla in datos_planta:
            planta.append(numpy.array(tupla)[0]*100)
        humedad_por_plantas.append(planta)

    return humedad_por_plantas

def obtener_dato(sensor_tupla, dato_tupla , datos_por_planta):
    dato_indice_por_plantas = []

    for datos_planta in datos_por_planta:
        planta = []
        for tuplas in datos_planta:
            for tupla in tuplas:
                if(tupla[1] % sensor_tupla == 0):
                    planta.append(tupla[dato_tupla])
                    break
        dato_indice_por_plantas.append(planta)

    return dato_indice_por_plantas

def crear_graficas(datos_por_plantas,pred_por_plantas,indicePlanta,dias,fecha_datos_plantas,indice_1,indice_2,titulo):
    global pagina_abierta

    nombres = []
    diferencia_humedad_plantas = []

    for i in range(len(datos_por_plantas)):
        diferencia_humedad_plantas.append(abs(datos_por_plantas[i][-1] - pred_por_plantas[i][-1]))
        nombres.append(f"Planta {i+1}")

    for i in range(len(datos_por_plantas)):
        if diferencia_humedad_plantas[i] >= alterta_diferencia_humedad_roja:
            nombres[i] = nombres[i] + "🔴"
        elif diferencia_humedad_plantas[i] >= alterta_diferencia_humedad_amarilla:
            nombres[i] = nombres[i] + "🟡"
        else:
            nombres[i] = nombres[i] + "🟢"

    fig = make_subplots(rows=num_lineas_pagina, cols=num_columnas_pagina, subplot_titles=nombres)

    color_verde = 'rgb(46, 204, 113)'  # Verde
    color_azul = 'rgb(52, 152, 219)'  # Azul

    linea = 1
    columna = 1
    for i in range(len(datos_por_plantas)):

        if linea == num_lineas_pagina + 1:
            columna = columna + 1
            linea = 1

        fecha_datos = fecha_datos_plantas[i]
        dato_por_planta = datos_por_plantas[i]
        pred_por_planta = pred_por_plantas[i]

        fig.add_trace(go.Scatter(x=fecha_datos, y=dato_por_planta, mode='lines', name=indice_1, marker=dict(color = color_azul)), row = linea, col = columna)
        fig.add_trace(go.Scatter(x=fecha_datos, y=pred_por_planta, mode='lines', name=indice_2, marker=dict(color = color_verde)), row = linea, col = columna)

        linea = linea + 1

    # Personalizar el diseño del gráfico
    fig.update_layout(
        title=titulo,
        xaxis=dict(title='Fecha de los datos'),
        yaxis=dict(title='Humedad'),
        showlegend=True,
        hovermode='closest'
    )

    #Tabla resumen
    tabla = go.Figure(data=[go.Table(
        header=dict(values=['Estado plantas']),  # Reemplaza estos valores con tus propios encabezados
        cells=dict(values=[nombres])
        # Reemplaza estos valores con tus propios datos
    )])

    tabla.write_html('tabla_plantas.html')

    fig.write_html('graficas.html')

    insertar_tabla_en_pagina_graficas()

    if not(pagina_abierta):
        pagina_abierta = True
        abrir_html_en_navegador()

def abrir_html_en_navegador():
    webbrowser.open_new_tab('graficas.html')  # Abre una nueva pestaña para evitar cerrar la anterior
    # Esto es JavaScript para recargar la página automáticamente
    script = f"""
    <script>
        setTimeout(function() {{
            window.location.reload(true);
        }}, {tiempo_entre_busquedas * 60 * 1000});
    </script>
    """
    with open('graficas.html', 'a') as f:
        f.write(script)

def insertar_tabla_en_pagina_graficas():
    # Leer el contenido de tabla_plantas.html con codificación utf-8
    with open('tabla_plantas.html', 'r', encoding='utf-8') as tabla_file:
        tabla_content = tabla_file.read()

    # Leer el contenido de graficas.html
    with open('graficas.html', 'r', encoding='utf-8') as graficas_file:
        graficas_content = graficas_file.read()

    # Encontrar el índice donde insertar la tabla
    insert_index = graficas_content.find('</body>')

    # Insertar la tabla al final de graficas.html
    nuevo_contenido = graficas_content[:insert_index] + tabla_content + graficas_content[insert_index:]

    # Escribir el nuevo contenido en graficas.html
    with open('graficas.html', 'w', encoding='utf-8') as graficas_file:
        graficas_file.write(nuevo_contenido)

def recoger_datos_nuevos():
    global datos_nuevos
    print("Recogiendo datos...")

    try:
        conexion = psycopg2.connect(database='PlantasIA', user='postgres', password="@Andriancito2012@")
        cursor = conexion.cursor()

        comando = f'''
SELECT * FROM "NuevosDatosPlantasIA"
    where id >= {ultimo_id}
ORDER BY device_id, date, signal_id ASC

        '''
        """
SELECT * FROM public."NuevosDatosPlantasIA"
	where date >= '2024-02-19 12:30:00' and date <= '2024-03-11 12:00:00'
ORDER BY device_id, date, signal_id ASC

SELECT * FROM "NuevosDatosPlantasIA"
    where id >= {ultimo_id}
ORDER BY device_id, date, signal_id ASC

SELECT * FROM public."NuevosDatosPlantasIA"
    WHERE device_id > 0 and device_id < 6
ORDER BY device_id, date, signal_id ASC
        """

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
    global num_columnas_pagina
    global num_lineas_pagina

    print("Tratando datos...")

    datos_por_planta = dividir_datos_por_planta(datos_nuevos)

    if(len(datos_por_planta) % num_columnas_pagina == 0):
        num_lineas_pagina = int(len(datos_por_planta)/num_columnas_pagina)
    else:
        num_lineas_pagina = int(len(datos_por_planta) / num_columnas_pagina) + 1

    fecha_datos_plantas = obtener_dato(1,4,datos_por_planta)

    X_IOT = eliminar_datos_inecesarios(datos_por_planta)

    #Tenemos un array con tantos array como plantas donde hay arrays de tuplas
    X_normalizado = preparar_datos_normalizados_red(X_IOT)

    humedad_por_plantas = obtener_humedad(X_normalizado)

    print("Realizando predicciones...")

    for planta_normalizada in X_normalizado:
        prediccion_por_planta.append(prediccion_plantas(planta_normalizada))


    print("Creando graficas...")

    indice_humedad = 'Humedad planta'
    indice_predicion = 'Prediccion IA'
    titulo = 'Predicciones realizadas por el modelo frente a la humedad real de la planta'

    crear_graficas(humedad_por_plantas, prediccion_por_planta,0,0,fecha_datos_plantas,indice_humedad,indice_predicion,titulo)

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
        print(f"Esperando {tiempo_entre_busquedas} minutos...")
        time.sleep(tiempo_entre_busquedas * 60)
