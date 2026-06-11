from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import bcrypt
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BASE_PUBLIC_URL = "http://127.0.0.1:8000"

DB_CONFIG = {
    'host': "localhost",
    'user': "root",
    'password': "root123",
    'database': "db_kayumanis"
}

# Load Model
MODEL_PATH = 'model_kayu_manis_mobilenetv2.h5'   

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil di upload!")
except Exception as e:
    print(f"Gagal upload model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print("DB Error:", e)
        return None

def get_quality_label(class_index):
    labels = {0: "Baik", 1: "Sedang", 2: "Buruk"}
    return labels.get(class_index, "Unknown")


@app.route('/')
def home():
    return jsonify({"message": "Backend Klasifikasi Kualitas Kayu Manis"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    required = ['username', 'email', 'password']
    
    if not all(key in data for key in required):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Cek username & email
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", 
                      (data['username'], data['email']))
        if cursor.fetchone():
            return jsonify({'status': 'error', 'message': 'Username or email already exists'}), 400

        cursor.execute('''
            INSERT INTO users (username, email, password)
            VALUES (%s, %s, %s)
        ''', (data['username'], data['email'], hashed_password))
        
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'status': 'success', 'message': 'Registrasi berhasil'}), 201

    except Error as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not all(key in data for key in ['username', 'password']):
        return jsonify({'status': 'error', 'message': 'Missing fields'}), 400

    username = data['username']
    password = data['password'].encode('utf-8')

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, username, email, password 
            FROM users 
            WHERE username = %s OR email = %s
        """, (username, username))
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
            return jsonify({
                'status': 'success',
                'message': 'Login berhasil',
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email']
                }
            }), 200
        else:
            return jsonify({'status': 'error', 'message': 'Username atau password salah'}), 401

    except Error as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image part'}), 400

    image = request.files['image']
    user_id = request.form.get('user_id')

    if not user_id:
        return jsonify({'status': 'error', 'message': 'user_id diperlukan'}), 400

    if image.filename == '' or not allowed_file(image.filename):
        return jsonify({'status': 'error', 'message': 'Image tidak valid'}), 400

    # Simpan gambar
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{secure_filename(image.filename)}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    image.save(file_path)

    file_url = f"{BASE_PUBLIC_URL}/uploads/{filename}"

    # Prediksi Model
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][class_index])
        quality_name = get_quality_label(class_index)
    except Exception as e:
        return jsonify({
        'status': 'error',
        'message': str(e)
    }), 500

    # Simpan ke Database
    conn = get_db_connection()
    description = "Tidak ada deskripsi dari server."
    if conn:
        cursor = conn.cursor()
        try:
            # Insert ke images
            cursor.execute("INSERT INTO images (user_id, image_url) VALUES (%s, %s)", (user_id, file_url))
            image_id = cursor.lastrowid

            # Ambil quality_id
            cursor.execute("SELECT id, description FROM quality_category WHERE quality_name = %s", (quality_name,))
            quality = cursor.fetchone()

            if quality:
                quality_id = quality[0]
                description = quality[1]
            else:
                quality_id = 1

            cursor.execute('''
                INSERT INTO classification_history 
                (user_id, image_id, quality_id, confidence)
                VALUES (%s, %s, %s, %s)
            ''', (user_id, image_id, quality_id, confidence))
            
            conn.commit()
        except Error as e:
            print("DB Save Error:", e)
        finally:
            cursor.close()
            conn.close()

    return jsonify({
        'status': 'success',
        'class': quality_name,
        'confidence': round(confidence * 100, 2),
        'description': description,
        'image_url': file_url
    }), 200

@app.route('/api/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500

    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT h.id, q.quality_name, h.confidence, h.detected_at, i.image_url
        FROM classification_history h
        JOIN quality_category q ON h.quality_id = q.id
        JOIN images i ON h.image_id = i.id
        WHERE h.user_id = %s
        ORDER BY h.detected_at DESC
    """
    cursor.execute(query, (user_id,))
    history = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify({'status': 'success', 'data': history})

@app.route('/api/history/detail/<int:history_id>', methods=['GET'])
def history_detail(history_id):

    conn = get_db_connection()

    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT 
            h.id,
            h.confidence,
            q.quality_name,
            q.description,
            i.image_url
        FROM classification_history h
        JOIN quality_category q ON h.quality_id = q.id
        JOIN images i ON h.image_id = i.id
        WHERE h.id = %s
    """

    cursor.execute(query, (history_id,))
    data = cursor.fetchone()

    cursor.close()
    conn.close()

    if not data:
        return jsonify({
            'status': 'error',
            'message': 'History tidak ditemukan'
        }), 404

    return jsonify({
        'status': 'success',
        'data': data
    })

@app.route('/api/history/delete/<int:history_id>', methods=['DELETE'])
def delete_history(history_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500
    
    try: 
        cursor = conn.cursor()
        cursor.execute("SELECT image_id FROM classification_history WHERE id = %s", (history_id,)) 
        row = cursor.fetchone()

        if not row:
            return jsonify({'status': 'error', 'message': 'History tidak ditemukan'}), 404

        image_id = row[0]

        cursor.execute("DELETE FROM classification_history WHERE id = %s", (history_id,))
        cursor.execute("DELETE FROM images WHERE id = %s", (image_id,))

        conn.commit() 
        return jsonify({'status': 'success', 'message': 'History berhasil dihapus'})

    except Error as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        cursor.close()
        conn.close()    

@app.route('/api/quality-categories', methods=['GET'])
def get_quality_categories():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, quality_name, description FROM quality_category")
    categories = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify({'status': 'success', 'data': categories})

@app.route('/api/profile/update', methods=['PUT'])
def update_profile():
    data = request.get_json()

    # Pastikan frontend mengirimkan user_id
    if 'user_id' not in data:
        return jsonify({'status': 'error', 'message': 'user_id diperlukan'}), 400

    user_id = data['user_id']
    updates = []
    params = []

    if 'username' in data and data['username'].strip():
        updates.append("username = %s")
        params.append(data['username'].strip())

    if 'email' in data and data['email'].strip():
        updates.append("email = %s")
        params.append(data['email'].strip())

    if 'password' in data and data['password'].strip():
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        updates.append("password = %s")
        params.append(hashed_password)

    if not updates:
        return jsonify({'status': 'error', 'message': 'Tidak ada data yang diupdate'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Cek username/email baru sudah digunakan user lain
        if 'username' in data or 'email' in data:
            check_query = "SELECT id FROM users WHERE (username = %s OR email = %s) AND id != %s"
            check_params = (data.get('username', ''), data.get('email', ''), user_id)
            cursor.execute(check_query, check_params)
            if cursor.fetchone():
                return jsonify({'status': 'error', 'message': 'Username atau email sudah digunakan user lain'}), 400

        query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
        params.append(user_id)

        cursor.execute(query, tuple(params))
        conn.commit()

        # mengambil data baru user
        cursor.execute("SELECT id, username, email FROM users WHERE id = %s", (user_id,))
        updated_user = cursor.fetchone()

        cursor.close()
        conn.close()

        return jsonify({
            'status': 'success',
            'message': 'Profile berhasil diperbarui',
            'user': updated_user
        }), 200

    except Error as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
# Upload Files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)