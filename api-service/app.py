from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import traceback
import mysql.connector
from datetime import datetime
import pytz
import requests

app = Flask(__name__)
CORS(app) 

# Kunci utamanya ada di sini. URL ini akan diambil dari environment variable di Railway
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL') 

# Konfigurasi database juga diambil dari environment variable
db_config = {
    'host': os.environ.get('MYSQLHOST'),
    'user': os.environ.get('MYSQLUSER'),
    'password': os.environ.get('MYSQLPASSWORD'),
    'database': os.environ.get('MYSQLDATABASE'),
    'port': os.environ.get('MYSQLPORT')
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    if not ML_SERVICE_URL:
        return jsonify({'error': 'ML service URL is not configured'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(save_path)

    try:
        # Mengirim gambar ke layanan ML
        with open(save_path, 'rb') as f:
            files = {'image': (image_file.filename, f, image_file.mimetype)}
            response = requests.post(ML_SERVICE_URL, files=files)
            response.raise_for_status()
        
        result = response.json()

        # Menyimpan hasil dari layanan ML ke database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (filename, classification, accuracy, drynessLevel, is_banana) VALUES (%s, %s, %s, %s, %s)",
            (result['filename'], result['classification'], result['accuracy'], result['drynessLevel'], result['is_banana'])
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'ML service is unavailable: {e}'}), 503
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

# Endpoint /history dan /uploads tetap sama persis seperti kode Anda sebelumnya
@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM history ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        jakarta = pytz.timezone('Asia/Jakarta')
        for row in rows:
            if isinstance(row['created_at'], datetime):
                row['created_at'] = row['created_at'].astimezone(jakarta).strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': 'Failed to fetch history'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=8000)