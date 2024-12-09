import numpy as np  # Librería para manipulación de datos numéricos
from flask import Flask, render_template, request  # Flask para crear la aplicación web
from tensorflow.keras.preprocessing import image  # Preprocesamiento de imágenes para modelos TensorFlow
from tensorflow.keras.models import load_model  # Cargar modelos entrenados
from PIL import Image, ImageFile  # Librería Pillow para manipular imágenes
import my_tf_mod  # Módulo personalizado para preprocesamiento y predicción
from io import BytesIO  # Trabajar con datos binarios en memoria
import matplotlib  # Librería para crear gráficas
matplotlib.use('Agg')  # Configuración para usar Matplotlib en aplicaciones web (sin GUI)
import matplotlib.pyplot as plt  # Generar gráficas
import base64  # Codificación/decodificación de datos en formato Base64

# Inicializa la aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    """
    Ruta principal de la aplicación.

    Renderiza la página de inicio ('home.html') que contiene un formulario
    para subir una imagen.
    """
    return render_template('home.html')  # Renderizar el archivo HTML de la página principal


@app.route('/Prediction', methods=['GET', 'POST'])
def pred():
    """
    Ruta para realizar predicciones.

    - Procesa una imagen subida por el usuario.
    - Clasifica la fruta (manzana, plátano, naranja).
    - Determina si la fruta está fresca o podrida.
    - Muestra la imagen subida junto con los resultados de la predicción.
    """
    if request.method == 'POST':  # Verificar si la solicitud es de tipo POST
         file = request.files['file']  # Obtener el archivo subido por el usuario
         org_img, img = my_tf_mod.preprocess(file)  # Preprocesar la imagen para los modelos

         print(img.shape)  # Imprimir las dimensiones de la imagen preprocesada
         fruit_dict = my_tf_mod.classify_fruit(img)  # Clasificar la fruta en una categoría
         rotten = my_tf_mod.check_rotten(img)  # Determinar si está fresca o podrida

         # Crear una imagen visualizable en la página web
         img_x = BytesIO()  # Crear un buffer de memoria para almacenar la imagen
         plt.imshow(org_img)  # Mostrar la imagen original
         plt.axis('off')  # Quitar los ejes de la gráfica
         plt.savefig(img_x, format='png')  # Guardar la imagen en formato PNG en el buffer
         plt.close()  # Cerrar la figura para liberar memoria
         img_x.seek(0)  # Mover el cursor del buffer al inicio
         plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')  # Convertir la imagen a Base64 para incrustarla en HTML

    # Renderizar la plantilla con los resultados
    return render_template(
        'Pred3.html', 
        fruit_dict=fruit_dict,  # Diccionario con probabilidades de clasificación
        rotten=rotten,  # Lista con probabilidades de frescura
        plot_url=plot_url  # Imagen en formato Base64 para mostrarla en la página web
    )


# Punto de entrada principal para ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)  # Ejecutar la aplicación Flask en modo debug
