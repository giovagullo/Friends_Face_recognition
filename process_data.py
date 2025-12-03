import os
import cv2
import numpy as np

# Rutas base 
BASE_PATH = '/home/giovanni/Desktop/Deep Learning/PROYECTO_FINAL/dataset' # Para "generalizar" las rutas
SETS = ['train', 'test', 'valid']
IMG_SIZE = (128, 128)

def crear_tensores(split_name): #split_name es 'train', 'test' o 'valid', basicamente porque las distintas imagenes estan en distintas carpetas
    """
    Lee imágenes y etiquetas, recorta las caras y devuelve arrays X e y.
    """
    path_images = os.path.join(BASE_PATH, split_name, 'images') #Ruta de las imagenes
    path_labels = os.path.join(BASE_PATH, split_name, 'labels')
    
    if not os.path.exists(path_images):
        print(f"No se encontró la carpeta: {path_images}")
        return None, None

    x_data = [] #Esto se va a llenar con las imagenes procesadas, y se va a devolver como un array numpy, es lo que antes en MNIST y CIFAR10 era X_train, X_test, etc
    y_data = [] #Esto se va a llenar con las etiquetas correspondientes a las imagenes, y se va a devolver como un array numpy, es lo que antes en MNIST y CIFAR10 era y_train, y_test, etc
    
    archivos = [f for f in os.listdir(path_images) if f.endswith(('.jpg', '.png', '.jpeg'))] #Esto lee todos los archivos de imagen en la carpeta de imagenes
    print(f"Procesando {split_name}: {len(archivos)} imágenes encontradas...")

    for archivo in archivos:
        # 1. Leer imagen
        ruta_img = os.path.join(path_images, archivo) #Ruta completa de la imagen, combinando la carpeta y el nombre del archivo
        img = cv2.imread(ruta_img)
        if img is None: continue
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Esto es porque OpenCV lee en BGR, mientras que la mayoria de librerias usan RGB
        h_img, w_img, _ = img.shape #Altura y ancho de la imagen original
        # 2. Leer etiqueta correspondiente
        # Asume que la etiqueta tiene el mismo nombre pero con extensión .txt
        nombre_txt = os.path.splitext(archivo)[0] + ".txt"
        ruta_txt = os.path.join(path_labels, nombre_txt)
        
        if os.path.exists(ruta_txt): #Todo esto es para leer las etiquetas en formato YOLO, es deir donde estan las caras en la imagen
            with open(ruta_txt, 'r') as f:
                lineas = f.readlines()
                for linea in lineas:
                    partes = linea.strip().split()

                    #El formato YOLO es: <clase> <x_centro> <y_centro> <ancho> <alto> (todo normalizado 0-1)
                    clase_id = int(partes[0]) 
                    x_center, y_center = float(partes[1]), float(partes[2])
                    ancho, alto = float(partes[3]), float(partes[4])
                    
                    # 3. Convertir coordenadas normalizadas a píxeles (ROI) 
                    x1 = int((x_center - ancho / 2) * w_img)
                    y1 = int((y_center - alto / 2) * h_img)
                    x2 = int((x_center + ancho / 2) * w_img)
                    y2 = int((y_center + alto / 2) * h_img)
                    
                    # Corregir límites (por si el recorte se sale de la imagen)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    # Recortar la cara
                    cara = img[y1:y2, x1:x2]
                    
                    if cara.size == 0: continue # Evitar recortes vacíos
                    try:
                        cara_resized = cv2.resize(cara, IMG_SIZE) # Redimensionar a 128x128 o el tamaño que queramos usar en la CNN
                        # Normaliza aca a 0-1
                        cara_norm = cara_resized.astype('float32') / 255.0 
                        
                        x_data.append(cara_norm)
                        y_data.append(clase_id)
                    except Exception as e:
                        print(f"Error procesando {archivo}: {e}")

    return np.array(x_data), np.array(y_data)

# EJECUCIÓN PARA LOS DISTINTOS SETS 
for split in SETS:
    print(f"--- Generando tensores para: {split} ---")
    X, y = crear_tensores(split)
    
    if X is not None and len(X) > 0:
        # Guardar en disco como archivos .npy
        np.save(os.path.join(BASE_PATH, f'X_{split}.npy'), X)
        np.save(os.path.join(BASE_PATH, f'y_{split}.npy'), y)
        
        print(f"Guardado: X_{split}.npy con forma {X.shape}")
        print(f"Guardado: y_{split}.npy con forma {y.shape}")
    else:
        print(f"Advertencia: No se generaron datos para {split}")

print("\n¡Proceso terminado! Tensores listos para entrenar.")

#%%
import matplotlib.pyplot as plt

# esto es solo para visualizar que los datos se cargaron bien
BASE_PATH = '/home/giovanni/Desktop/Deep Learning/PROYECTO_FINAL/dataset'
class_names = ['Monica', 'Chandler', 'Phoebe', 'Rachel', 'Joey', 'Ross']

# 1. Cargar los datos procesados
print("Cargando tensores...")
X_train = np.load(f'{BASE_PATH}/X_train.npy')
y_train = np.load(f'{BASE_PATH}/y_train.npy')

print(f"Dataset cargado: {X_train.shape} imágenes, {y_train.shape} etiquetas")

# 2. Visualizar muestras aleatorias
plt.figure(figsize=(12, 6))
indices = np.random.randint(0, len(X_train), 10) # Elegir 10 al azar

for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    
    # X_train está normalizado (0-1), matplotlib lo muestra directo
    plt.imshow(X_train[idx]) 
    
    # Obtener el nombre usando tu lista
    label_numerico = int(y_train[idx])
    nombre = class_names[label_numerico]
    
    plt.title(f"{label_numerico}: {nombre}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('muestras_procesadas.jpg')
plt.show()
# %%
