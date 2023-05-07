# from flask import Flask, request, jsonify
# import os
# from tqdm import tqdm

# app = Flask(__name__)

# @app.route('/virtualtry')
# def virtualtry():
#     return "Hello"

# @app.route('/')
# def out_recomms():
#     return "Home"

from flask import Flask, request, jsonify, send_from_directory
import os
from tqdm import tqdm
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return 1
#     except:
#         return 0

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=11, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

app = Flask(__name__)

@app.route('/virtualtry')
def virtualtry():
    return "Hello"

@app.route('/', methods=['GET'])
def out_recomms():
    in_img_path = os.path.join('img', request.args.get('in_img_path'))
    model_run = request.args.get('model_run')
    if model_run == 'yes':
        filenames = []
        for file in os.listdir('img'):
            filenames.append(os.path.join('img', file))
        feature_list = []
        for file in tqdm(filenames):
            feature_list.append(extract_features(file, model))
        pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
        pickle.dump(filenames,open('filenames.pkl','wb'))

    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    out_img_paths = []
    features = feature_extraction(in_img_path, model)
    indices = recommend(features, feature_list)
    
    for i in range(10):
        row = {}
        row['img_path'] = os.path.basename(filenames[indices[0][i]])
        out_img_paths.append(row)
    return jsonify(out_img_paths)

@app.route("/img/<path:imgPath>")
def send_img(imgPath):
    return send_from_directory('img', imgPath)

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="5000")
    app.run(host="192.168.43.233", port="5000")
