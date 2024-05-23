import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import glob
import collections
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import  locale

from keras.src.saving import register_keras_serializable
locale.setlocale(locale.LC_ALL,'es_ES')

from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers

#Wilcoxon Test
import warnings
warnings.filterwarnings('ignore')

import cv2

import psycopg2
from psycopg2 import Error

#Entrenamiento
batch_size = 64
epochs = 100
crossValidationSplit = 10
busquedaMejorTuplaDias = 7
validation_split = 0.10

planta_imagen = 1
img_rows, img_cols = 64, 64

# Datos en entrada en la red
input_shape_IOT = (9,)

nombreModelo = "modelo1_32x32"

#Variables Globales
resultadosLossModelos = []
modelos = []
datosIOT = []
datosIOTEntrenamientoImagenes = []


###################
# Modelo imagenes #
###################

def load_data():
    X = []
    fechas = []

    print("-Procensado imagenes")

    for filename in glob.glob(f'../NDVIfotos/*.png'):
        # Cambiamos nombre de NDVI_2024-02-19_12_32_44 a 2024-02-19_12_32
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
        datosIOTEntrenamientoImagenes.append(tupla_con_fecha_mas_cercana(fechasImagenes[i], datosIOT[planta_imagen - 1]))
    return datosIOTEntrenamientoImagenes

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

def asignarHumedadPerfectaImagenes(planta):
    global humedadIdealOrdenadaPorPlantas

    print(f"Planta {planta_imagen-1} procensando...")
    humedadIdealOrdenadaPorPlantas = obtenerHumedadPerfecta(planta)

    return humedadIdealOrdenadaPorPlantas

#######################
# Modelo sin imagenes #
#######################
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

    #ultima planta
    datos_por_planta_desorganizados.append(datos_planta)

    for datos_planta in datos_por_planta_desorganizados:
        datos_por_planta.append(juntar_datos_planta(datos_planta))

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
            contador = contador + 1
            anterior_sensor = datos[2]
            lecturaCompleta.append(datos)
            if contador == 16:
                lectura_ordenada = sorted(lecturaCompleta, key=lambda x: x[2])
                datos_organizados.append(lectura_ordenada)
                lecturaCompleta = []
                contador = 0

    return datos_organizados
def asignarHumedadPerfecta(plantas):
    humedadIdealOrdenadaPorPlantas = []

    for planta in plantas:
        humedadIdealOrdenadaPorPlantas.append(obtenerHumedadPerfecta(planta))

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

    return humedadPerfectaPlanta

def crearGraficasHumedad(humedadPerfectaPlanta,humedadMasGrandeDato,indicePlanta,dias,fechaDeCadaDato,indice_1,indice_2,titulo):
    plt.plot(humedadPerfectaPlanta, label=indice_1, marker='o', color='blue')  # Puntos de array1
    plt.plot(humedadMasGrandeDato, label=indice_2, marker='o', color='red')  # Puntos de array2

    # Añadir título y etiquetas
    plt.title(titulo)
    plt.xlabel('Indice array')
    plt.ylabel('Humedad')

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

def preparar_datos_normalizados_red(X,y):
    # Inicializamos un nuevo array para almacenar todas las tuplas
    datos_por_planta_normalizados = []

    max_X = [100,40,3492.0, -1,690.0,1641.0,1646.0,1920.0,1746.0]

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
    #print("Por planta")
    #for max_planta in max_X_por_planta:
    #    print(max_planta)
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


    #######
    #Normalizamos la humedad
    humedad_normalizada = []
    # Iteramos sobre cada array interno en y
    for humedad_planta in y:
        humedades_planta = []
        # Iteramos sobre cada tupla dentro del array interno y las normalizamos
        for elemento in humedad_planta:
            normalized_tupla = elemento / 100
            humedades_planta.append(normalized_tupla)
        humedad_normalizada.append(np.array(humedades_planta))


    return datos_por_planta_normalizados, humedad_normalizada

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

def cnn_model_IOT(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64,  activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(1,activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

#########
# Comun #
#########
def obtenerDatosIOT():
    datosIOT = []
    print("-Procensado datos IOT")
    try:
        conexion = psycopg2.connect(database = 'PlantasIA', user = 'postgres', password = "@Andriancito2012@")
        cursor = conexion.cursor()
        print("Extrayendo Datos:")
        comando = '''
SELECT * 
FROM public."DatosIOT"
WHERE date <= '2024-03-05 23:59:59' 
  AND device_id IN (1, 2, 3, 4)
ORDER BY device_id, date, signal_id ASC; 
'''
        cursor.execute(comando)
        datosIOT = cursor.fetchall()
    except Error as e:
        print("Error en la conexion: ",e)

    return datosIOT

@register_keras_serializable()
def custom_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss

def GuardarValoresROCfichero():
    # Escribimos los resultado en un fichero de texto
    with open(f"datosROC{nombreModelo}.txt", 'w') as file:
        # Escribir los elementos en el archivo, separados por espacios
        file.write(' '.join(map(str, resultadosROC)))

##################################################################################
# Mains program
##################################################################################

################
# Sin Imagenes #
################
def mainSinImagenes():
    global resultadosROC
    print("Recogiendo datos:")
    datosIOT = obtenerDatosIOT()

    datos_por_planta = dividir_datos_por_planta(datosIOT)

    X_sin_normalizar= eliminar_datos_inecesarios(datos_por_planta)

    y_sin_normalizar = asignarHumedadPerfecta(datos_por_planta)

    X_por_planta,y_por_planta = preparar_datos_normalizados_red(X_sin_normalizar,y_sin_normalizar)

    #Ponemos tanto X como y en un mismo array
    X = []
    y = []

    for planta in X_por_planta:
        for dato in planta:
            X.append(dato)

    for planta in y_por_planta:
        for dato in planta:
            y.append(dato)

    X = np.array(X)
    y = np.array(y)

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

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=2)
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

        #Visualizar datos del split
        loss = model.evaluate(X_test, y_test, batch_size=batch_size)
        y_pred = model.predict(X_test)

        resultadosLossModelos.append(loss[1])

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

        modelos.append(model)

    loss_mas_bajo = 1
    contador = 0
    for loss in resultadosLossModelos:
        if loss < loss_mas_bajo:
            loss_mas_bajo = loss
            modelos[contador].save(f'greentwin_{contador}.keras')

        contador = contador + 1


    print(loss_mas_bajo)
    #model.save('greentwin.keras')

################
# Con Imagenes #
################
def mainImagenes():
    print("Recogiendo datos:")
    X_imagenes, fechas, input_shape = load_data()

    fechasImagenes = obtenerFechaImagenesFormatoSQL(fechas)

    """
    #Mostrar imagenes
    print(X_IOT.shape, 'train samples')
    print(img_rows, 'x', img_cols, 'image size')
    print(input_shape, 'input_shape')
    print(epochs, 'epochs')
    """

    datosIOT = obtenerDatosIOT()

    datos_por_planta = dividir_datos_por_planta(datosIOT)

    datosFiltrados = filtrarDatosIOTparaImagenes(fechasImagenes, datos_por_planta)

    X_sin_normalizar = eliminar_datos_inecesarios(datosFiltrados)

    y_sin_normalizar = asignarHumedadPerfecta(datosFiltrados)

    X_por_planta, y_por_planta = preparar_datos_normalizados_red(X_sin_normalizar, y_sin_normalizar)

    # Ponemos tanto X como y en un mismo array
    X = []
    y = []

    for planta in X_por_planta:
        for dato in planta:
            X.append(dato)

    for planta in y_por_planta:
        for dato in planta:
            y.append(dato)

    X = np.array(X)
    y = np.array(y)

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

    model.save('greentwin_img.keras')

if __name__ == '__main__':
    entrenar_con_imagen = False

    if entrenar_con_imagen:
        mainImagenes()
    else:
        mainSinImagenes()
