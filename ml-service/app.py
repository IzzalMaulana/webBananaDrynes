from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import pickle
import os
import traceback
from transformers import ViTImageProcessor, ViTModel
import torch

app = Flask(__name__)

# Memuat semua model berat di sini
MIN_CONFIDENCE = 76.0 

try:
    model_path = 'model_xgboost_pisang_result.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("XGBoost model loaded!")
except Exception as e:
    model = None

try:
    device = torch.device('cpu')
    extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    vit_model.eval()
    vit_available = True
    print("ViT model loaded!")
except Exception as e:
    vit_available = False

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

def get_prediction_confidence(features):
    probabilities = model.predict_proba(features)
    return float(np.max(probabilities) * 100)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vit_available:
        return jsonify({'error': 'Model is not available'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    try:
        features = preprocess_image(image_file.read())
        confidence = get_prediction_confidence(features)

        if confidence < MIN_CONFIDENCE:
            result = {
                'classification': 'Gambar Bukan Pisang', 'accuracy': round(confidence, 1),
                'drynessLevel': -1, 'is_banana': False, 'filename': image_file.filename
            }
        else:
            pred = int(model.predict(features)[0])
            label_map = {0: "Basah", 1: "Sedang", 2: "Kering"}
            result = {
                'classification': label_map.get(pred, "Unknown"), 'accuracy': round(confidence, 1),
                'drynessLevel': pred, 'is_banana': True, 'filename': image_file.filename
            }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    app.run(port=8001)