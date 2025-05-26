from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import torch
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pdam.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Pelanggan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    alamat = db.Column(db.String(200), nullable=False)

class PencatatanMeteran(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pelanggan_id = db.Column(db.Integer, db.ForeignKey('pelanggan.id'), nullable=False)
    tanggal = db.Column(db.String(20), nullable=False)
    angka_meteran = db.Column(db.String(20), nullable=False)
    gambar_path = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

@app.route('/tambah_pelanggan', methods=['POST'])
def tambah_pelanggan():
    data = request.json
    pelanggan_baru = Pelanggan(nama=data['nama'], alamat=data['alamat'])
    db.session.add(pelanggan_baru)
    db.session.commit()
    return jsonify({'message': 'Pelanggan berhasil ditambahkan'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'pelanggan_id' not in request.form or 'tanggal' not in request.form:
        return jsonify({'error': 'Data tidak lengkap'}), 400

    pelanggan_id = request.form['pelanggan_id']
    tanggal = request.form['tanggal']
    image_file = request.files['image']

folder = 'uploads'
os.makedirs(folder, exist_ok=True)
image_path = os.path.join(folder, image_file.filename)
image_file.save(image_path)

image = Image.open(image_path)
results = model(image)
predictions = results.pandas().xyxy[0]

detected_numbers = []
for _, row in predictions.iterrows():
    detected_numbers.append(row['name'])

detected_number = ''.join(detected_numbers) if detected_numbers else "0"

pencatatan = PencatatanMeteran(
    pelanggan_id=pelanggan_id,
    tanggal=tanggal,
    angka_meteran=detected_number,
    gambar_path=image_path
)
db.session.add(pencatatan)
db.session.commit()
