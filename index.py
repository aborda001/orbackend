from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from models import openPoseModel, mediaPipeModel

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = ''
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['model'] = '0'

@app.route('/')
def index():
    model = request.args.get('model', default='0', type=str)
    app.config['model'] = model
    return "Hello, World!"


@app.route('/video/<filename>')
def video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():

    if 'video' not in request.files:
        return jsonify({'error': 'No selected file'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        if (app.config['model'] == '0'):
            result = mediaPipeModel(filepath)
        else:
            result = openPoseModel(filepath)


        return jsonify( {
            'message': 'File uploaded successfully',
            'newVideo': result['video'],
            'isVideoValid': result['is_valid']
        }), 200


if __name__ == '__main__':
    app.run(
        debug=True,
        port=5000
    )