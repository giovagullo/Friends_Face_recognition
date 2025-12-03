import os
#DESACTIVAR GPU PARA TENSORFLOW  Esto evita el conflicto de memoria con YOLO (PyTorch)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import keras

# Rutas de los modelos
YOLO_PATH = "/home/giovanni/yolo-face/yolov8n-face.pt"
CNN_PATH  = '/home/giovanni/Desktop/Deep Learning/PROYECTO_FINAL/dataset/mejor_modelo.h5'

# Ruta de la imagen NUEVA a probar
IMAGE_PATH = '/home/giovanni/Desktop/Deep Learning/albertengou.jpg' 

# Nombres de las clases     
CLASS_NAMES = ['Monica', 'Chandler', 'Phoebe', 'Rachel', 'Joey', 'Ross']

# Tamaño cnn input
IMG_SIZE = (128, 128) 

# Margen extra para el recorte 
MARGIN = 20 

def predict_on_full_image():
    # 1. Cargar Modelos
    print(f"Cargando YOLO desde: {YOLO_PATH}")
    yolo = YOLO(YOLO_PATH)
    
    print(f"Cargando CNN desde: {CNN_PATH}")
    try:
        cnn = keras.models.load_model(CNN_PATH)
    except Exception as e:
        print(f"Error cargando CNN: {e}")
        return

    # 2. Cargar Imagen Original
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: No se pudo leer la imagen en {IMAGE_PATH}")
        return
    
    # Convertir a RGB para mostrar con Matplotlib al final (OpenCV usa BGR)
    img_rgb_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h_orig, w_orig = img.shape[:2]
    print(f"Imagen cargada: {w_orig}x{h_orig}")

    # 3. Detectar caras con YOLO
    results = yolo.predict(img, conf=0.4, verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("No se detectaron caras en la imagen.")
        return

    print(f"Se detectaron {len(results[0].boxes)} caras.")

    # 4. Procesar cada cara detectada
    for box in results[0].boxes:
        # A. Obtener Coordenadas y Recortar 
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Aplicar margen 
        x1_c = max(0, x1 - MARGIN)
        y1_c = max(0, y1 - MARGIN)
        x2_c = min(w_orig, x2 + MARGIN)
        y2_c = min(h_orig, y2 + MARGIN)
        
        face_crop = img[y1_c:y2_c, x1_c:x2_c]
        
        if face_crop.size == 0:
            continue

        # 1. Convertir a RGB (Crucial: Tu entrenamiento usó cvtColor)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # 2. Resize a 128x128
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        
        # 3. NORMALIZAR 
        face_norm = face_resized.astype("float32") / 255.0
        
        # 4. Expandir dimensiones para Keras 
        face_batch = np.expand_dims(face_norm, axis=0)
        
        # C. PREDICCIÓN 
        probs = cnn.predict(face_batch, verbose=0)
        pred_idx = np.argmax(probs)
        confidence = np.max(probs) * 100
        pred_name = CLASS_NAMES[pred_idx]
        
        print(f"Cara en ({x1},{y1}): {pred_name} ({confidence:.1f}%)")

        # D. DIBUJAR (Visualización)
        # Color del recuadro (Verde Matrix)
        color = (0, 255, 0) 
        
        # Variables para texto
        font_scale = 1.0
        thickness = 2
        label = f"{pred_name} {confidence:.1f}%"
        
        # Dibujar rectángulo en la cara
        cv2.rectangle(img_rgb_viz, (x1, y1), (x2, y2), color, 3)
        
        # Fondo del texto adaptable
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_rgb_viz, (x1, y1 - h_text - 10), (x1 + w_text, y1), color, -1)
        
        # Texto
        cv2.putText(img_rgb_viz, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # 5. Mostrar
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb_viz)
    plt.axis('off')
    plt.title("Resultado Final")
    plt.tight_layout()
    plt.savefig('resultado_foto_albertengou.png')
    plt.show()

if __name__ == "__main__":
    predict_on_full_image()