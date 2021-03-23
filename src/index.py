import datetime
import os
import random

from flask import Flask, render_template, request, jsonify
from flask import Response
from werkzeug.utils import secure_filename

from classic_modern_classifier import cm_classifier

app = Flask(__name__)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg', 'gif')


def create_uuid():
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    random_num = random.randint(0, 100)
    if random_num <= 10:
        random_num = str(0) + str(random_num)
    unique_num = str(now_time) + str(random_num)
    return unique_num


def allowed_file(filename):
    if not isinstance(filename, str):
        return False
    filename = filename.lower()
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/hello')
def hello():
    res = Response('Hi, Python')
    res.headers['Content-Type'] = 'text/plain'
    return res


@app.route('/')
def index():
    res = Response('Hi, Python')
    res.headers['Content-Type'] = 'text/plain'
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['upload_file']
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        print(filename)
        ext = filename.rsplit('.', 1)[1]
        new_filename = create_uuid() + '.' + ext
        new_filepath = os.path.join(file_dir, new_filename)
        f.save(new_filepath)
        score, _class = cm_classifier.predict(new_filepath)
        return jsonify({'data': {'score': score, 'class': _class}, 'msg': 'predict success'})
    else:
        return jsonify({'type': None, 'msg': 'predict fail'})


if __name__ == '__main__':
    app.run('0.0.0.0', 5001, threaded=False)
