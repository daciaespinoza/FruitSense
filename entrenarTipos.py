import os
import cv2
import numpy as np
from tqdm import tqdm  # Barra de progreso
from random import shuffle  # Barajar datos
from sklearn.model_selection import train_test_split  # Dividir datos en conjuntos de entrenamiento y validación
from tensorflow.keras.utils import to_categorical  # Convertir etiquetas a formato categórico
from tensorflow.keras.models import Sequential, load_model  # Modelos secuenciales de Keras
from tensorflow.keras.layers import (Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense)  # Capas de red neuronal
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint  # Callbacks para el entrenamiento
from tensorflow.keras.layers import Input  # Capa de entrada para el modelo
import tensorflow.keras as keras
import matplotlib.pyplot as plt  # Graficar datos del entrenamiento

# Definición de la clase ClasificadorFrutas
class ClasificadorFrutas:
    def __init__(self, ruta_entrenamiento, ruta_prueba):
        """
        Inicializa las rutas de los conjuntos de datos.

        Args:
            ruta_entrenamiento (str): Ruta al conjunto de entrenamiento.
            ruta_prueba (str): Ruta al conjunto de prueba.
        """
        self.ruta_entrenamiento = ruta_entrenamiento
        self.ruta_prueba = ruta_prueba
        self.modelo = None  # Inicializa el modelo como None

    def cargar_datos(self, tipo="prueba"):
        """
        Carga y preprocesa los datos de imágenes de frutas.

        Esta función lee las imágenes desde la carpeta especificada (entrenamiento o prueba),
        las redimensiona, las convierte a formato RGB y las asocia a su etiqueta correspondiente.
        Luego, los datos se barajan y se dividen en imágenes (X) y etiquetas (Y).

        Args:
            tipo (str): Especifica si cargar datos de "entrenamiento" o "prueba".

        Returns:
            tuple: Arreglos X (imágenes) y Y (etiquetas).
        """
        calidad = ['apples', 'banana', 'oranges']  # Categorías de frutas
        X, Y, datos = [], [], []  # Inicializa listas para las imágenes, etiquetas y datos combinados
        ruta = self.ruta_prueba if tipo == "prueba" else self.ruta_entrenamiento  # Selecciona la ruta según el tipo

        # Iterar sobre cada categoría de frutas
        for categoria in tqdm(os.listdir(ruta)):  # Barra de progreso para los directorios
            etiqueta = calidad.index(next((q for q in calidad if q in categoria), None))  # Determina la etiqueta
            ruta_categoria = os.path.join(ruta, categoria)  # Ruta de la categoría

            for nombre_imagen in os.listdir(ruta_categoria):  # Iterar sobre las imágenes de la categoría
                img = cv2.imread(os.path.join(ruta_categoria, nombre_imagen))  # Leer imagen
                img = cv2.resize(img, (100, 100))  # Redimensionar a 100x100 píxeles
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
                datos.append([img, etiqueta])  # Agregar imagen y etiqueta a la lista de datos

        print('Barajando los datos...')
        shuffle(datos)  # Mezclar aleatoriamente los datos

        for imagen, etiqueta in datos:  # Separar imágenes y etiquetas
            X.append(imagen)
            Y.append(etiqueta)

        return np.array(X), np.array(Y)  # Convertir listas a arreglos de NumPy

    def construir_modelo(self):
        """
        Construye y compila el modelo de clasificación.

        Este modelo está diseñado para clasificar imágenes de frutas en tres categorías: manzanas,
        bananas y naranjas. Incluye varias capas convolucionales, normalización por lotes, pooling,
        Dropout y capas densas.

        Returns:
            None
        """
        modelo = Sequential([  # Modelo secuencial
            Input(shape=(100, 100, 3)),  # Capa de entrada para imágenes de 100x100 con 3 canales (RGB)
            Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),  # Capa convolucional
            BatchNormalization(),  # Normalización por lotes
            SeparableConv2D(32, (3, 3), padding='same', activation='relu'),  # Convolución separable
            MaxPooling2D((2, 2)),  # Submuestreo (pooling) para reducción de tamaño
            BatchNormalization(),
            Dropout(0.3),  # Regularización con Dropout
            SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
            SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
            SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.5),
            Flatten(),  # Aplanar las características para conectarlas a la capa densa
            Dense(128, activation='relu', kernel_initializer='he_uniform'),  # Capa completamente conectada
            Dropout(0.3),
            Dense(3, activation='softmax')  # Capa de salida con 3 clases (una por categoría)
        ])

        modelo.compile(
            loss=keras.losses.categorical_crossentropy,  # Función de pérdida
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Optimizador Adam
            metrics=['accuracy']  # Métricas a monitorear
        )
        self.modelo = modelo  # Asignar el modelo construido

    def entrenar_modelo(self, X, Y, X_val, Y_val, epochs=40, batch_size=20):
        """
        Entrena el modelo con los datos proporcionados.

        Esta función utiliza los datos de entrenamiento y validación para entrenar el modelo.
        Además, incluye callbacks para reducir la tasa de aprendizaje y guardar el mejor modelo.

        Args:
            X (numpy array): Imágenes de entrenamiento.
            Y (numpy array): Etiquetas de entrenamiento.
            X_val (numpy array): Imágenes de validación.
            Y_val (numpy array): Etiquetas de validación.
            epochs (int): Número de épocas para entrenar.
            batch_size (int): Tamaño del batch.

        Returns:
            History: Historial del entrenamiento.
        """
        lr_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=0.00002, cooldown=2)  # Callback para reducir el learning rate
        check_point = ModelCheckpoint(filepath='//wsl.localhost/Ubuntu/home/dacia/IAProyecto/modelo/local_fruit.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')  # Guardar el mejor modelo

        X, X_val = X / 255.0, X_val / 255.0  # Normalizar imágenes a rango [0, 1]
        Y, Y_val = to_categorical(Y), to_categorical(Y_val)  # Convertir etiquetas a formato categórico

        history = self.modelo.fit(X, Y, batch_size=batch_size, validation_data=(X_val, Y_val), epochs=epochs, callbacks=[lr_rate, check_point])  # Entrenar el modelo
        return history  # Retornar historial de entrenamiento

    def evaluar_modelo(self, X, Y):
        """
        Evalúa el modelo con los datos proporcionados.

        Esta función mide el rendimiento del modelo en términos de pérdida y precisión
        utilizando un conjunto de datos de prueba.

        Args:
            X (numpy array): Imágenes de evaluación.
            Y (numpy array): Etiquetas de evaluación.

        Returns:
            list: Resultados de la evaluación [pérdida, precisión].
        """
        X = X / 255.0  # Normalizar imágenes
        Y = to_categorical(Y)  # Convertir etiquetas a categóricas
        resultados = self.modelo.evaluate(X, Y)  # Evaluar modelo
        print("Resultados de la evaluación:", resultados)  # Imprimir resultados
        return resultados  # Retornar resultados

    def graficar_historial(self, history):
        """
        Genera gráficas del historial de entrenamiento.

        Esta función muestra gráficas que representan la evolución de la pérdida y precisión
        durante el entrenamiento.

        Args:
            history (History): Historial del entrenamiento.
        """
        plt.figure(1, figsize=(20, 12))  # Crear figura para las gráficas
        plt.subplot(1, 2, 1)  # Gráfica de la pérdida
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
        plt.plot(history.history["val_loss"], label="Pérdida de validación")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)  # Gráfica de la precisión
        plt.xlabel("Épocas")
        plt.ylabel("Precisión")
        plt.plot(history.history["accuracy"], label="Precisión de entrenamiento")
        plt.plot(history.history["val_accuracy"], label="Precisión de validación")
        plt.legend()
        plt.grid(True)
        plt.show()

    def guardar_modelo(self, ruta):
        """
        Guarda el modelo entrenado en la ruta especificada.

        Args:
            ruta (str): Ruta donde se guardará el modelo.
        """
        self.modelo.save(ruta)  # Guardar el modelo
        print(f"Modelo guardado en {ruta}")  # Confirmar la operación

# Ejemplo de uso
clasificador = ClasificadorFrutas(
    r"\\wsl.localhost\Ubuntu\home\dacia\ProyectoIA\dataset\train",  # Ruta del conjunto de entrenamiento
    r"\\wsl.localhost\Ubuntu\home\dacia\ProyectoIA\dataset\test"  # Ruta del conjunto de prueba
)

X_test, Y_test = clasificador.cargar_datos(tipo="prueba")  # Cargar datos de prueba
X, Y = clasificador.cargar_datos(tipo="entrenamiento")  # Cargar datos de entrenamiento

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)  # Dividir datos en entrenamiento y validación
clasificador.construir_modelo()  # Construir modelo

historia = clasificador.entrenar_modelo(X_train, Y_train, X_val, Y_val, epochs=50)  # Entrenar el modelo
clasificador.guardar_modelo('//wsl.localhost/Ubuntu/home/dacia/ProyectoIA/modelo/local_fruit_final.keras')  # Guardar modelo final
