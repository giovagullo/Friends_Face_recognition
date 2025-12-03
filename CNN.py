#%%
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
#%%
# Rutas y configuración
BASE_PATH = '/home/giovanni/Desktop/Deep Learning/PROYECTO_FINAL/dataset'
CLASS_NAMES = ['Monica', 'Chandler', 'Phoebe', 'Rachel', 'Joey', 'Ross']
NUM_CLASSES = len(CLASS_NAMES)

#%%

# ============================================================
# PASO 1: CARGA DE DATOS
# ============================================================
X_train = np.load(f'{BASE_PATH}/X_train.npy')
y_train = np.load(f'{BASE_PATH}/y_train.npy')

X_valid = np.load(f'{BASE_PATH}/X_valid.npy')
y_valid = np.load(f'{BASE_PATH}/y_valid.npy')

X_test = np.load(f'{BASE_PATH}/X_test.npy')
y_test = np.load(f'{BASE_PATH}/y_test.npy')

print(f"Train: {X_train.shape} imágenes, {y_train.shape} etiquetas")
print(f"Valid: {X_valid.shape} imágenes, {y_valid.shape} etiquetas")
print(f"Test:  {X_test.shape} imágenes, {y_test.shape} etiquetas")
#%%

# ============================================================
# PASO 2: ONE-HOT ENCODING
# ============================================================

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_valid = keras.utils.to_categorical(y_valid, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#%%

# ============================================================
# PASO 3: CONSTRUCCIÓN DE LA CNN
# ============================================================


model = Sequential([
    #Bloque convolucional 1, espera la entrada de 128x128 x 3 canales (RGB), y devuelve 32 filtros del mismo tamaño, luego esta la activacion ReLU y un maxpooling para reducir la dimensionalidad a la mitad
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(128, 128, 3), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    
    # Bloque convolucional 2, espera una entrada de 64x64 x 32 canales (después del primer maxpooling), y devuelve 64 filtros, luego activación ReLU y otro maxpooling para reducir la dimensionalidad a la mitad
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    
    # Bloque convolucional 3, espera una entrada de 32x32 x 64 canales, y devuelve 128 filtros, luego activación ReLU y otro maxpooling para reducir la dimensionalidad a la mitad
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
    layers.MaxPooling2D((2, 2), name='pool3'),
    
    # Bloque convolucional 4, espera una entrada de 16x16 x 128 canales, y devuelve 256 filtros, luego activación ReLU y otro maxpooling para reducir la dimensionalidad a la mitad
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
    layers.MaxPooling2D((2, 2), name='pool4'),
    #HASTA ACA TENEMOS UNA SALIDA DE 8x8 X 256 CANALES, UNA IMAGEN MUY REDUCIDA PERO CON MUCHOS FILTROS ES DECIR MUCHA INFORMACIÓN
    
    #AHORA lo aplanamos para poder conectarlo a las capas densas y hacer la clasificación (es decir ese tensor 3D de 8x8 x256 lo convertimos en un vector 1D de 8*8*256=16384 elementos)
    layers.Flatten(name='flatten'),
    
    # Capa oculta densa, con 256 neuronas a la salida y activación ReLU, seguida de Dropout para evitar overfitting, lo que hace es apagar aleatoriamente el 50% de las neuronas en cada época de entrenamiento
    layers.Dense(256, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout1'),  # ¡CRÍTICO para evitar overfitting!
    
    # Segunda capa oculta densa, con 128 neuronas y activación ReLU, seguida de otro Dropout del 50%
    layers.Dense(128, activation='relu', name='dense2'),
    layers.Dropout(0.5, name='dropout2'),
    
    # Ultima capa de salida, con tantas neuronas como clases (6 en este caso) y activación softmax para clasificación multi-clase
    layers.Dense(NUM_CLASSES, activation='softmax', name='output')
])

# Mostrar arquitectura
model.summary()

#%%

# ============================================================
# PASO 4: COMPILACIÓN DEL MODELO
# ============================================================

#Aca definimos el optimizador, la función de pérdida y las métricas que vamos a usar durante el entrenamiento
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#%%

# ============================================================
# PASO 5: CALLBACKS (Early Stopping y Model Checkpoint)
# ============================================================
# Estos callbacks ayudan a mejorar el entrenamiento deteniendolo si se estanca y guardar el mejor modelo

# Early Stopping: Detiene el entrenamiento si val_loss no mejora
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model Checkpoint: Guarda el mejor modelo
checkpoint = ModelCheckpoint(
    f'{BASE_PATH}/mejor_modelo.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


#%%


# ============================================================
# PASO 6: ENTRENAMIENTO CON DATA AUGMENTATION
# ============================================================


# 1. Crear el generador de datos (Solo para Train)
datagen = ImageDataGenerator(
    rotation_range=20,      # Rotar imagen aleatoriamente 20 grados
    width_shift_range=0.1,  # Mover horizontalmente 10%
    height_shift_range=0.1, # Mover verticalmente 10%
    shear_range=0.1,        # Inclinar
    zoom_range=0.1,         # Zoom in/out
    horizontal_flip=True,   # Espejo (Fundamental para caras)
    fill_mode='nearest'
)

# 2. Entrenar usando el generador
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=50,          
    restore_best_weights=True
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32), #En vez de pasar X_train e y_train directamente, usamos el generador para evitar que el modelo vea siempre las mismas imágenes
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) // 32,
    epochs=250, # Ponemos más épocas, el Augmentation hace que aprenda más lento pero mejor
    callbacks=[early_stop, checkpoint],
    verbose=1
)

#%%

# ============================================================
# PASO 7: VISUALIZACIÓN DE CURVAS DE APRENDIZAJE
# ============================================================


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Evolución del Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico de Accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Evolución del Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/curvas_entrenamiento.png', dpi=300)
plt.show()

print("Gráficos guardados en 'curvas_entrenamiento.png'")

#%%

# ============================================================
# PASO 8: EVALUACIÓN EN TEST SET
# ============================================================


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

#%%

# ============================================================
# PASO 9: PREDICCIONES Y MATRIZ DE CONFUSIÓN
# ============================================================

# Obtener predicciones
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)  # Clase con mayor probabilidad

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Cantidad de predicciones'})
plt.title('Matriz de Confusión - Test Set', fontsize=16, fontweight='bold')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/matriz_confusion.png', dpi=300)
plt.show()

print("Matriz de confusión guardada en 'matriz_confusion.png'")

# ============================================================
# PASO 10: REPORTE DE CLASIFICACIÓN
# ============================================================


report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
print(report)

# Guardar reporte en archivo de texto
with open(f'{BASE_PATH}/reporte_clasificacion.txt', 'w') as f:
    f.write("REPORTE DE CLASIFICACIÓN - CNN FRIENDS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

print("\n¡Proceso completado exitosamente!")
print(f"Archivos generados en: {BASE_PATH}/")
print("  - mejor_modelo.keras")
print("  - curvas_entrenamiento.png")
print("  - matriz_confusion.png")
print("  - reporte_clasificacion.txt")
# %%
# ============================================================
# PASO 11: GUARDADO FINAL DEL MODELO
# ============================================================

model.save(f'{BASE_PATH}/mejor_modelo.h5')

print(f"Modelo guardado exitosamente en: {BASE_PATH}/mejor_modelo.h5")
# %%
