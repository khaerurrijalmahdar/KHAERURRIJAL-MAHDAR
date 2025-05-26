import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Izinkan akses dari frontend

# Konfigurasi database SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/datameteranair'
db = SQLAlchemy(app)

# Model Database
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    image_path = db.Column(db.String(200), nullable=True)
    detected_numbers = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Buat database jika belum ada
with app.app_context():
    db.create_all()

# Load Model YOLOv8
model = YOLO("runs/detect/train/weights/best.pt")  # Sesuaikan path model

# Endpoint untuk menampilkan halaman utama (opsional)
@app.route('/')
def home():
    return render_template('index.html')

# API untuk menambahkan pengguna
@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    new_user = UserData(name=data['name'], email=data['email'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User added successfully'}), 201

# API untuk mendapatkan daftar pengguna
@app.route('/users', methods=['GET'])
def get_users():
    users = UserData.query.all()
    user_list = [{'id': user.id, 'name': user.name, 'email': user.email,
                  'image_path': user.image_path, 'detected_numbers': user.detected_numbers,
                  'created_at': user.created_at} for user in users]
    return jsonify(user_list)

# API untuk unggah gambar dan deteksi angka
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    user_id = request.form.get('user_id')

    if not file or not user_id:
        return jsonify({'error': 'File or user_id is missing'}), 400

    filename = f"user_{user_id}_{file.filename}"
    file_path = os.path.join("static/uploads", filename)
    file.save(file_path)

    # Jalankan YOLOv8 untuk deteksi angka
    results = model(file_path)

    detected_numbers = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Ambil class ID
            detected_numbers.append(str(cls_id))

    detected_text = "".join(detected_numbers)

    # Simpan hasil ke database
    user = UserData.query.get(user_id)
    if user:
        user.image_path = file_path
        user.detected_numbers = detected_text
        db.session.commit()

    return jsonify({'message': 'File uploaded successfully', 'detected_numbers': detected_text}), 200

# Jalankan aplikasi Flask
if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True)  # Buat folder upload jika belum ada
    app.run(debug=True)
