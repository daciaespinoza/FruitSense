from tensorflow.keras.models import load_model  # Importar para cargar modelos entrenados
import numpy as np  # Librería para operaciones numéricas
from tensorflow.keras.preprocessing import image  # Utilidades para preprocesar imágenes
from tensorflow.keras.preprocessing.image import img_to_array  # Convertir imágenes a arrays NumPy
from PIL import Image, ImageFile  # Librería para manejar imágenes (Pillow)
from io import BytesIO  # Leer archivos binarios desde memoria

# Cargar el modelo que evalúa la calidad de la fruta (fresco o podrido)
quality_model = load_model('//wsl.localhost/Ubuntu/home/dacia/ProyectoIA/modelo/rottenvsfresh.keras')
# Cargar el modelo que clasifica la fruta en categorías (manzana, plátano, naranja)
clf_model = load_model('//wsl.localhost/Ubuntu/home/dacia/ProyectoIA/modelo/local_fruit_final.keras')


def preprocess(file):
    """
    Preprocesa la imagen para hacerla compatible con los modelos.
    
    - Convierte la imagen a formato RGB.
    - Normaliza los valores de píxeles entre 0 y 1.
    - Redimensiona la imagen a 100x100 píxeles para los modelos.
    
    Args:
        file (File): Archivo de imagen cargado.

    Returns:
        tuple: 
            - org_img_array: Array de la imagen original normalizado.
            - img: Imagen procesada lista para el modelo (redimensionada y normalizada).
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # Prevenir errores al leer imágenes incompletas
    org_img = Image.open(BytesIO(file.read())).convert('RGB')  # Abrir imagen y convertir a formato RGB
    org_img.load()  # Cargar completamente la imagen en memoria

    # Convertir la imagen original a un array NumPy y normalizar
    org_img_array = image.img_to_array(org_img)  # Convertir la imagen original a un array NumPy
    org_img_array = org_img_array.astype('float32') / 255.0  # Normalizar los valores de píxeles a [0, 1]

    # Redimensionar la imagen a 100x100 píxeles para los modelos
    img = org_img.resize((100, 100), Image.Resampling.LANCZOS)  # Redimensionar usando LANCZOS para mayor calidad
    img = image.img_to_array(img)  # Convertir a un array NumPy
    img = img.astype('float32') / 255.0  # Normalizar los valores de píxeles a [0, 1]
    img = np.expand_dims(img, axis=0)  # Agregar una dimensión para simular un lote de imágenes

    return org_img_array, img  # Retornar la imagen original y la preprocesada


def check_rotten(img):
    """
    Evalúa si la fruta está fresca o podrida.

    Utiliza el modelo `quality_model` para predecir las probabilidades
    de que la fruta esté fresca o podrida.

    Args:
        img (numpy array): Imagen preprocesada lista para el modelo.

    Returns:
        list: [probabilidad_fresco (%), probabilidad_podrido (%)].
    """
    # Predecir la probabilidad de frescura y calcular la de podrido como complemento
    return [
        round(100 * quality_model.predict(img)[0][0], 3),  # Probabilidad de fresco
        round(100 * (1 - quality_model.predict(img)[0][0]), 3)  # Probabilidad de podrido
    ]

def classify_fruit(img):
    """
    Clasifica la fruta en una de las categorías disponibles.

    Utiliza el modelo `clf_model` para predecir las probabilidades
    de que la fruta pertenezca a las categorías: manzana, plátano o naranja.

    Args:
        img (numpy array): Imagen preprocesada lista para el modelo.

    Returns:
        dict: Diccionario con las probabilidades por categoría en porcentaje.
    """
    fru_dict = {}  # Diccionario para almacenar las probabilidades por categoría

    # Predecir las probabilidades para cada categoría
    fru_dict['apple'] = round(clf_model.predict(img)[0][0] * 100, 4)  # Probabilidad de manzana
    fru_dict['banana'] = round(clf_model.predict(img)[0][1] * 100, 4)  # Probabilidad de plátano
    fru_dict['orange'] = round(clf_model.predict(img)[0][2] * 100, 4)  # Probabilidad de naranja

    # Ajustar probabilidades muy pequeñas a cero para evitar ruido
    for value in fru_dict:
        if fru_dict[value] <= 0.001:
            fru_dict[value] = 0.00

    return fru_dict  # Retornar el diccionario con las probabilidades
