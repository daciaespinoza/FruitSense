import tensorflow as tf  # Librería principal para construir y entrenar modelos de aprendizaje profundo
import numpy as np  # Librería para manipulación de datos numéricos
import os  # Utilidades para manejar rutas y archivos del sistema
import cv2  # Librería para procesamiento de imágenes
import matplotlib.pyplot as plt  # Librería para visualización de datos
from tqdm import tqdm  # Barra de progreso para loops largos
from random import shuffle  # Utilidad para mezclar datos aleatoriamente
from sklearn.model_selection import train_test_split  # Dividir datos en conjuntos de entrenamiento y validación
from tensorflow.keras.utils import to_categorical  # Para convertir etiquetas a formato categórico
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D, Activation,
                                      Flatten, BatchNormalization, SeparableConv2D)  # Capas para construir redes neuronales
from tensorflow.keras.models import Sequential  # Modelo secuencial para redes neuronales

class ClasificadorFrutas:
    def __init__(self, ruta_datos_entrenamiento, ruta_datos_prueba):
        """
        Inicializa las rutas del conjunto de datos y el modelo.

        Args:
            ruta_datos_entrenamiento (str): Ruta a los datos de entrenamiento.
            ruta_datos_prueba (str): Ruta a los datos de prueba.
        """
        self.ruta_entrenamiento = ruta_datos_entrenamiento
        self.ruta_prueba = ruta_datos_prueba
        self.modelo = None  # El modelo será inicializado más adelante

    def cargar_imagenes_muestra(self, max_imagenes=6):
        """
        Carga una muestra limitada de imágenes desde el conjunto de entrenamiento.

        Args:
            max_imagenes (int): Número máximo de imágenes por categoría.

        Returns:
            numpy array: Arreglo de imágenes.
        """
        imagenes = []
        for subdirectorio in tqdm(os.listdir(self.ruta_entrenamiento)):  # Iterar sobre las carpetas de categorías
            ruta_subdirectorio = os.path.join(self.ruta_entrenamiento, subdirectorio)
            for i, nombre_imagen in enumerate(os.listdir(ruta_subdirectorio)):  # Iterar sobre las imágenes de la carpeta
                if i >= max_imagenes:  # Limitar el número de imágenes cargadas
                    break
                img = cv2.imread(os.path.join(ruta_subdirectorio, nombre_imagen))  # Leer la imagen
                img = cv2.resize(img, (100, 100))  # Redimensionar a 100x100 píxeles
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
                imagenes.append(img)
        return np.array(imagenes)

    def mostrar_imagenes(self, imagenes, titulos=None):
        """
        Muestra un conjunto de imágenes en una cuadrícula con títulos opcionales.

        Args:
            imagenes (list): Lista de imágenes a mostrar.
            titulos (list, opcional): Lista de títulos para las imágenes.
        """
        if len(imagenes) == 36:  # Comprobar si hay suficientes imágenes para una cuadrícula de 6x6
            fig, axes = plt.subplots(6, 6, figsize=(20, 20))
            for i, img in enumerate(imagenes):
                ax = axes[i // 6, i % 6]  # Asignar la imagen a una posición en la cuadrícula
                ax.imshow(img)
                ax.axis('off')  # Ocultar los ejes
                if titulos is not None:
                    ax.set_title(titulos[i])  # Agregar título si está disponible
            plt.show()
        else:
            print("El número de imágenes no es suficiente para mostrar.")

    def cargar_datos(self, tipo="entrenamiento"):
        """
        Carga las imágenes y etiquetas del conjunto de datos especificado.

        Args:
            tipo (str): "entrenamiento" o "prueba".

        Returns:
            tuple: Arreglos de imágenes y etiquetas.
        """
        calidad = ['fresh', 'rotten']  # Clases: fresco y podrido
        imagenes, etiquetas = [], []
        datos = self.ruta_entrenamiento if tipo == "entrenamiento" else self.ruta_prueba  # Seleccionar el conjunto de datos

        for categoria in tqdm(os.listdir(datos)):  # Iterar sobre las categorías
            etiqueta = 0 if calidad[0] in categoria else 1  # Asignar etiqueta (0: fresco, 1: podrido)
            ruta_categoria = os.path.join(datos, categoria)

            for nombre_imagen in os.listdir(ruta_categoria):  # Iterar sobre las imágenes de la categoría
                img = cv2.imread(os.path.join(ruta_categoria, nombre_imagen))  # Leer la imagen
                img = cv2.resize(img, (100, 100))  # Redimensionar a 100x100 píxeles
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
                imagenes.append(img)
                etiquetas.append(etiqueta)

        shuffle(imagenes)  # Mezclar las imágenes aleatoriamente
        return np.array(imagenes), np.array(etiquetas)

    def construir_modelo(self):
        """
        Construye y compila el modelo de red neuronal.

        El modelo utiliza varias capas convolucionales, normalización por lotes,
        Dropout y capas densas para realizar la clasificación.
        """
        modelo = Sequential()
        modelo.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=(100, 100, 3)))
        modelo.add(BatchNormalization())
        modelo.add(SeparableConv2D(32, (3, 3), padding='same', activation='relu'))
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(0.3))

        modelo.add(SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
        modelo.add(SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(Dropout(0.4))

        modelo.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Flatten())
        modelo.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        modelo.add(Dropout(0.3))
        modelo.add(Dense(1, activation='sigmoid'))  # Salida binaria (fresco o podrido)

        modelo.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        self.modelo = modelo
        print(self.modelo.summary())  # Mostrar resumen del modelo

    def entrenar_modelo(self, X, Y, X_val, Y_val, epochs=50, batch_size=20):
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            X (numpy array): Datos de entrenamiento.
            Y (numpy array): Etiquetas de entrenamiento.
            X_val (numpy array): Datos de validación.
            Y_val (numpy array): Etiquetas de validación.
            epochs (int): Número de épocas.
            batch_size (int): Tamaño del batch.

        Returns:
            History: Historial del entrenamiento.
        """
        lr_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, mode='max', min_lr=0.00002, cooldown=2)
        check_point = tf.keras.callbacks.ModelCheckpoint(filepath='//wsl.localhost/Ubuntu/home/dacia/ProyectoIA/modelo/rotten.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        X, X_val = X / 255.0, X_val / 255.0  # Normalizar las imágenes

        history = self.modelo.fit(X, Y, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[lr_rate, check_point])
        return history

    def evaluar_modelo(self, X_val, Y_val):
        """
        Evalúa el modelo en el conjunto de datos de validación.

        Args:
            X_val (numpy array): Datos de validación.
            Y_val (numpy array): Etiquetas de validación.

        Returns:
            list: Resultados de la evaluación (pérdida y precisión).
        """
        X_val = X_val / 255.0  # Normalizar las imágenes
        resultados = self.modelo.evaluate(X_val, Y_val)
        print("Resultados de la evaluación:", resultados)
        return resultados

    def guardar_modelo(self, ruta):
        """
        Guarda el modelo entrenado en la ruta especificada.

        Args:
            ruta (str): Ruta donde se guardará el modelo.
        """
        self.modelo.save(ruta)
        print(f"Modelo guardado en {ruta}")

# Uso de la clase
clasificador = ClasificadorFrutas(
    ruta_datos_entrenamiento= r"\\wsl.localhost\Ubuntu\home\dacia\ProyectoIA\dataset\train", 
    ruta_datos_prueba= r"\\wsl.localhost\Ubuntu\home\dacia\ProyectoIA\dataset\test"
)

# Cargar los datos de prueba y entrenamiento
X_test, Y_test = clasificador.cargar_datos(tipo="prueba")
X, Y = clasificador.cargar_datos(tipo="entrenamiento")

# Dividir los datos en entrenamiento y validación
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Construir el modelo
clasificador.construir_modelo()

# Entrenar el modelo
historia = clasificador.entrenar_modelo(X_train, Y_train, X_val, Y_val, epochs=50)

# Guardar el modelo entrenado
clasificador.guardar_modelo('//wsl.localhost/Ubuntu/home/dacia/ProyectoIA/modelo/rottenvsfresh.keras')
