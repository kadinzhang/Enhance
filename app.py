from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from utils import load_image
from model import resolve_single
from PIL import Image

import sys
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
# sys.path.append(os.path.abspath('./weights'))
import model.common

# Load in model and weights
def init():
    json_file = open('weights/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("weights/gan_generator.h5")
    print("Loaded Model from disk")
    graph = tf.compat.v1.get_default_graph()
    return loaded_model,graph


global model, graph
model, graph = init()


UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == "POST":
        if request.files:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image = load_image(file_path)

            high_res_np = resolve_single(model, image)
            high_res = Image.fromarray(high_res_np.numpy(), 'RGB')
            high_res.save(os.path.join(OUTPUT_FOLDER, filename))

            return render_template('results.html', filename=filename)

# Make the images available in templates
@app.route('/outputs/<filename>')
def send_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/uploads/<filename>')
def send_upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)