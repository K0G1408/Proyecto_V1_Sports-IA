#!/usr/bin/env python3

# Práctica 02: Redes Neuronales Convolucionales.

# Importamos librerías y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout

# Creamos la CNN
classifier = Sequential()

# Definimos capas de convolución
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))

# Aplicamos una normalización Batch
classifier.add(BatchNormalization())

# Reducimos dimensionalidad de los resultados
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Creamos una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Hacemos un aplanado al mapa de características
classifier.add(Flatten())

# Definimos capas densas
classifier.add(Dense(units = 128, activation = "relu"))

# Aplicamos capas Dropout
classifier.add(Dropout(0.5))

# Finalizamos red neuronal con 10 neuronas (pues hay 10 clases)
# y una función de activación softmax
classifier.add(Dense(units = 10, activation = "softmax"))

# Compilamos la CNN con un optimizador adam, una función de pérdida de
# entropía cruzada categórica y usamos la exactitud como métrica de rendimiento del modelo.
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Preprocesamiento de datos 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, # Escalamos valores de la imagen entre 0 y 1
        shear_range=0.2, # Definimos un valor de transformación de corte de imágenes de 20%
        zoom_range=0.2, # Definimos un valor de zoom de imágenes del 20%
        horizontal_flip=True) # Permitimos rotaciones en las imágenes

test_datagen = ImageDataGenerator(rescale=1./255)

# Clasificación de imágenes
training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

history = classifier.fit_generator(training_dataset,
                        steps_per_epoch=150,
                        epochs=20,
                        validation_data=testing_dataset,
                        validation_steps=44)

#categories = ["ajedrez", "baloncesto", "boxeo", "disparo", "esgrima", "formula1",
              #"futbol", "hockey", "natacion", "tenis"]
 
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix

# Cargar y preprocesar la imagen para que coincida con el tamaño esperado por el modelo
rutas = ["single_test/ajedrez.jpg", "single_test/baloncesto.jpg", "single_test/boxeo.jpg",
         "single_test/disparo.jpg", "single_test/esgrima.jpg", "single_test/formula1.jpg", 
         "single_test/futbol.jpg", "single_test/hockey.jpg", "single_test/natacion.jpg", 
         "single_test/tenis.jpg"]
y_pred = []
y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for ruta in rutas:
    test_image = load_img(ruta, target_size=(64, 64)) # Cambia la ruta a tu imagen
    test_image = img_to_array(test_image)
    
    # Normalizar la imagen como en el entrenamiento
    test_image = test_image / 255.0
    
    # Expandir las dimensiones para hacer la predicción (1, 128, 128, 3)
    test_image = np.expand_dims(test_image, axis=0)
    
    # Predecir la clase
    result = classifier.predict(test_image) #Aqui va el nombre del modelo que crearon
    
    # Mostrar los valores predichos (probabilidades para cada clase)
    print("Probabilidades predichas:", result)
    
    # Obtener el índice de la clase con mayor probabilidad
    predicted_class_index = np.argmax(result)
    
    # Obtener el mapeo de clases
    class_labels = training_dataset.class_indices
    
    # Invertir el diccionario para obtener las clases por índice
    class_labels = dict((v, k) for k, v in class_labels.items())
    
    # Obtener el nombre de la clase predicha
    predicted_class_label = class_labels[predicted_class_index]
    
    # Añadimos el valor de la clase predicho para esta imagen
    y_pred.extend(np.argmax(result, axis=1) + 1)
    
    # Imprimir la clase predicha
    print(f'La imagen pertenece a la clase: {predicted_class_label}')

conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

import matplotlib.pyplot as plt
# Graficar la pérdida
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida durante el entrenamiento y la validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la precisión
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión durante el entrenamiento y la validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()