import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from flask import Flask, jsonify, request
from flask import send_from_directory
import mysql.connector
from mysql.connector import Error
import bcrypt

app = Flask(__name__)

# Konfigurasi Database
DB_CONFIG = dict(
    host="localhost",
    user="root",
    password="root123", 
    database="db_kayumanis", 
)

MODEL_PATH = 'model_kayu_manis_mobilenetv2.h5'

try: 
    model = load_model(MODEL_PATH)
    print("Model Kayu Manis berhasil diproses")
except Exception as e: 
    print(f"Gagal Proses model: {e}")

    class_names = ['Baik', 'Sedang', 'Buruk']

# untuk konek ke mysql
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print("Error koneksi database:", e)
        return None

@app.route('/', methods=['GET'])
def cek_server():
    conn = get_db_connection()
    if conn and conn.is_connected():
        status_db = "Berhasil terkoneksi ke database"
        conn.close() # Jangan lupa ditutup lagi pintunya
    else:
        status_db = "Gagal terkoneksi ke database."

    return jsonify({
        'status': 'success',
        'message': 'Welcome To Backend Klasifikasi KayuManis',
        'database': status_db
    }), 200

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    required_fields = ['username', 'email', 'password']
    if not all(key in data for key in required_fields):
        return jsonify({'error': 'Data tidak lengkap, pastikan semua data terisi'}), 400
    
    # Proses enkripsi password
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Pengecekan username duplikat
        cursor.execute('SELECT id FROM users WHERE username = %s', (data['username'],))
        if cursor.fetchone():
            return jsonify({'error': f"Username '{data['username']}' sudah digunakan"}), 400
        
        # Pengecekan email duplikat
        cursor.execute('SELECT id FROM users WHERE email = %s', (data['email'],))
        if cursor.fetchone():
            return jsonify({'error': f"Email '{data['email']}' sudah terdaftar"}), 400
        
        # Simpan ke tabel user
        cursor.execute('''
            INSERT INTO users (username, email, password)
            VALUES (%s, %s, %s)
        ''', (data['username'], data['email'], hashed_password))
        
        conn.commit()
        # Mengambil ID user yang baru saja dibuat
        new_id = cursor.lastrowid 
        cursor.close()
        conn.close()
        return jsonify({'id': new_id, 'message': 'Registrasi berhasil!'}), 201
        
    except Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not all(key in data for key in ['username', 'password']):
        return jsonify({'error': 'Username/Email dan password harus diisi'}), 400
    
    username = data['username'] 
    password = data['password'].encode('utf-8')
    
    try:
        conn = get_db_connection()
        # dictionary=True agar hasil query berupa dictionary (mudah diambil key-nya)
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s OR email = %s', (username, username))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user is None:
            return jsonify({'error': 'Username atau email tidak ditemukan'}), 401
        
        # Membandingkan password input dengan password di database
        if bcrypt.checkpw(password, user['password'].encode('utf-8')):
            return jsonify({
                'message': 'Login berhasil',
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email']
                }
            }), 200
        else:
            return jsonify({'error': 'Password salah'}), 401
            
    except Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID wajib disertakan'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400
    
    file = request.files['image']
    
    try:
        class_names = ['Baik', 'Sedang', 'Buruk'] 
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        import time
        filename = f"{int(time.time())}_{file.filename}"
        img_path = os.path.join("uploads", filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        result_class = class_names[np.argmax(score)]
        confidence = float(np.max(score)) * 100

       #simpan data ke database
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO images (user_id, image_url) 
            VALUES (%s, %s)
        ''', (user_id, img_path))
        
        image_id = cursor.lastrowid

        # id kualitas otomatis dari tabel quality_category
        cursor.execute('''
            SELECT id FROM quality_category WHERE quality_name = %s
        ''', (result_class,))
        quality_row = cursor.fetchone()
        
        # otomatis menjadi 1 jika ada data yang tidak sinkron
        quality_id = quality_row[0] if quality_row else 1 

        cursor.execute('''
            INSERT INTO classification_history (user_id, image_id, quality_id, confidence)
            VALUES (%s, %s, %s, %s)
        ''', (user_id, image_id, quality_id, confidence))
        
        conn.commit()
        cursor.close()
        conn.close()     

        return jsonify({
            'status': 'success',
            'quality': result_class,
            'confidence': f"{confidence:.2f}%",
            'image_url': f"http://127.0.0.1:8000/{img_path}" 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = '''
            SELECT 
                ch.id AS history_id,
                ch.confidence,
                ch.detected_at,
                i.file_path AS image_url,
                q.quality_name AS quality
            FROM classification_history ch
            JOIN images i ON ch.image_id = i.id
            JOIN qualities q ON ch.quality_id = q.id
            WHERE ch.user_id = %s
            ORDER BY ch.detected_at DESC
        '''
        
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        
        # file_path agar bisa dibuka di browser
        for row in rows:
            row['image_url'] = f"http://127.0.0.1:8000/{row['image_url']}"
            
        cursor.close()
        conn.close()
        return jsonify(rows), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/uploads/<filename>')
def get_uploaded_image(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)