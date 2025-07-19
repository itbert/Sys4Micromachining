from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import os
import uuid

from ipynb.fs.full.postprocessing import get_data  # айайай

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def predict_image(image_path):
    return get_data(image_path)  # айайай


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    try:
        img = Image.open(file.stream)

        unique_id = uuid.uuid4().hex
        bmp_filename = f"{unique_id}.bmp"
        bmp_path = os.path.join(app.config['UPLOAD_FOLDER'], bmp_filename)

        img.save(bmp_path, format='BMP')

        prediction = predict_image(bmp_path)

        txt_filename = f"result_{unique_id}.txt"
        txt_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

        with open(txt_path, 'w') as f:
            f.write(f"Prediction result: {prediction}\n")

        return send_file(txt_path, as_attachment=True)

    except Exception as e:
        return f"Error processing image: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
