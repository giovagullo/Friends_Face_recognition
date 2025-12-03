from flask import Flask, request, jsonify
from flask_cors import CORS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from ultralytics import YOLO
import keras

app = Flask(__name__)

# Configuración CORS más permisiva
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Cargar modelos al iniciar
YOLO_PATH = "/home/giovanni/yolo-face/yolov8n-face.pt"
CNN_PATH = '/home/giovanni/Desktop/Deep Learning/PROYECTO_FINAL/dataset/mejor_modelo.h5'
CLASS_NAMES = ['Monica', 'Chandler', 'Phoebe', 'Rachel', 'Joey', 'Ross']
IMG_SIZE = (128, 128)
MARGIN = 20

print("Cargando modelos...")
yolo = YOLO(YOLO_PATH)
cnn = keras.models.load_model(CNN_PATH)
print("Modelos cargados!")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Manejar preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400
    
    file = request.files['image']
    
    try:
        # Leer imagen
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Imagen inválida'}), 400
        
        h_orig, w_orig = img.shape[:2]
        
        # Detectar caras con YOLO
        results = yolo.predict(img, conf=0.4, verbose=False)
        
        faces_detected = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Aplicar margen
                x1_c = max(0, x1 - MARGIN)
                y1_c = max(0, y1 - MARGIN)
                x2_c = min(w_orig, x2 + MARGIN)
                y2_c = min(h_orig, y2 + MARGIN)
                
                face_crop = img[y1_c:y2_c, x1_c:x2_c]
                
                if face_crop.size == 0:
                    continue
                
                # Preprocesar
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, IMG_SIZE)
                face_norm = face_resized.astype("float32") / 255.0
                face_batch = np.expand_dims(face_norm, axis=0)
                
                # Predicción
                probs = cnn.predict(face_batch, verbose=0)
                pred_idx = np.argmax(probs)
                confidence = float(np.max(probs) * 100)
                pred_name = CLASS_NAMES[pred_idx]
                
                faces_detected.append({
                    'character': pred_name,
                    'confidence': round(confidence, 1),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        response = jsonify({
            'faces': faces_detected,
            'total_faces': len(faces_detected)
        })
        
        # Headers CORS adicionales
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Endpoint de prueba"""
    response = jsonify({'status': 'ok', 'message': 'Backend funcionando'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    from waitress import serve
    print("=" * 50)
    print("🚀 Servidor corriendo en http://localhost:5000")
    print("=" * 50)
    serve(app, host='0.0.0.0', port=5000, threads=4)