import numpy as np
import matplotlib.pyplot as plt
import keras
import random

# Rutas y configuración
BASE_PATH = '/home/giovanni/Desktop/Deep Learning/PROYECTO_FINAL/dataset'
MODEL_PATH = f'{BASE_PATH}/mejor_modelo.h5' # Ruta del modelo entrenado
CLASS_NAMES = ['Monica', 'Chandler', 'Phoebe', 'Rachel', 'Joey', 'Ross']

def visualizar_predicciones(num_samples=15):
    # 1. Cargar datos y modelo
    print("Cargando datos de Test y Modelo...")
    X_test = np.load(f'{BASE_PATH}/X_test.npy')
    y_test = np.load(f'{BASE_PATH}/y_test.npy') 
    
    # Si y_test viene en One-Hot (ej: [[0,1,0...]]), lo convertimos a número
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        print("Asegúrate de que el archivo .keras o .h5 existe en la ruta indicada.")
        return

    # 2. Seleccionar índices aleatorios
    indices = random.sample(range(len(X_test)), num_samples)
    
    # 3. Configurar la figura (Grid dinámico)
    cols = 5
    rows = (num_samples // cols) + (1 if num_samples % cols != 0 else 0)
    plt.figure(figsize=(15, 3.5 * rows))
    
    print(f"Visualizando {num_samples} imágenes aleatorias...")

    for i, idx in enumerate(indices):
        ax = plt.subplot(rows, cols, i + 1)
        
        # Obtener imagen y etiqueta real
        img = X_test[idx]
        true_label_index = y_test[idx]
        true_name = CLASS_NAMES[true_label_index]
        
        # Predecir (el modelo espera un batch, así que añadimos una dimensión extra)
        # img.shape es (128, 128, 3) -> pasamos a (1, 128, 128, 3)
        img_expanded = np.expand_dims(img, axis=0)
        prediction_probs = model.predict(img_expanded, verbose=0)
        pred_label_index = np.argmax(prediction_probs)
        pred_name = CLASS_NAMES[pred_label_index]
        confidence = np.max(prediction_probs) * 100
        
        # Determinar color (Verde si acertó, Rojo si falló)
        if true_label_index == pred_label_index:
            color = 'green'
            status = "OK"
        else:
            color = 'red'
            status = "ERROR"
            
        # Mostrar imagen
        # Nota: Si tus imágenes están normalizadas (0 a 1) o (0 a 255), matplotlib suele manejarlo bien.
        # Si se ven extrañas: ax.imshow(img.astype('uint8'))
        ax.imshow(img)
        
        # Dibujar el "Recuadro" (Borde de la gráfica)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3) # Grosor del recuadro
            
        # Título con la predicción
        ax.set_title(f"Real: {true_name}\nPred: {pred_name}\n({confidence:.1f}%)", 
                     color=color, fontweight='bold', fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualizar_predicciones(num_samples=15) 