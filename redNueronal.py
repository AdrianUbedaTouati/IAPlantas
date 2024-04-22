import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import collections
import matplotlib.pyplot as plt

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

#Entrenamiento
batch_size = 64
nb_classes = 3
epochs = 50
crossValidationSplit = 10
# Scaling input image to theses dimensions
img_rows, img_cols = 64, 64

nombreModelo = "modelo1_32x32"

resultadosROC = []

def load_data():
    X = []
    fecha = []
    for filename in glob.glob(f'./NDVIfotos/*.png'):
        # Dejamos de NDVI_2024-02-19_12_32_44 : 2024-02-19_12_32
        fecha.append(filename[17:-7])
        print(filename[17:-7])

        im = preprocesar_imagen(filename)
        X.append(image.img_to_array(im))

    input_shape = (img_rows, img_cols, 1)
    return np.array(X), np.array(fecha), input_shape

def preprocesar_imagen(imagen_path):
    # Cargar la imagen utilizando OpenCV
    imagen = cv2.imread(imagen_path)

    imagenRGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    imagenFinal = imagenRGB.astype('float32') / 255.0

    # Realizar otras transformaciones si es necesario, como redimensionar, normalizar, etc.
    #imagen_final = cv2.resize(imagen, (img_rows, img_cols))

    return imagenFinal

def plot_symbols(X,y,n=15):
    print(y)
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

def cnn_model(input_shape, nb_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
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

    plot_symbols(X, y)
    collections.Counter(y)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

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
