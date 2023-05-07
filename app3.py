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
import firebase_admin
from firebase_admin import credentials, storage, firestore
cred = credentials.Certificate("mygroceryapp-80ee5-594f1736fbc0.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'mygroceryapp-80ee5.appspot.com'}) # connecting to firebase
db = firestore.client()

from google.cloud import storage
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("mygroceryapp-80ee5-594f1736fbc0.json")


from flask import Flask, request, jsonify
# import os
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
    img = image.load_img(tensorflow.keras.utils.get_file(origin=img_path), target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def feature_extraction(img_path, model):
    img = image.load_img(tensorflow.keras.utils.get_file(origin=img_path), target_size=(224, 224))
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
    in_img_path = request.args.get('in_img_path')
    # in_img_path = 'https://firebasestorage.googleapis.com/v0/b/mygroceryapp-80ee5.appspot.com/o/img%2F' + request.args.get('in_img_path') + '?alt=media'
    # in_img_path = '2576.jpg'
    model_run = request.args.get('model_run')
    if model_run == 'yes':
        
        # storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob('img/10000.jpg').download_to_filename('10000n.jpg')

        files = storage.Client(credentials=credentials).list_blobs(firebase_admin.storage.bucket().name, prefix='img') # fetch all the files in the bucket
        # for i in files: print('The public url is ', i.public_url)

        # exit

        filenames = []
        for file in files:
            fil = file.public_url
            fil = fil.split("/")[-1]
            doc = db.collection('Products').where('filename', 'in', [fil])
            # print(doc == [])
            if fil != '' and doc != []:
                filenames.append('https://firebasestorage.googleapis.com/v0/b/mygroceryapp-80ee5.appspot.com/o/img%2F' + fil + '?alt=media')
                print(fil)
        feature_list = []
        for file in tqdm(filenames):
            feature_list.append(extract_features(file, model))
        pickle.dump(feature_list, open('embeddings2.pkl', 'wb'))
        pickle.dump(filenames,open('filenames2.pkl','wb'))

    feature_list = np.array(pickle.load(open('embeddings2.pkl', 'rb')))
    filenames = pickle.load(open('filenames2.pkl', 'rb'))
    out_img_paths = []
    features = feature_extraction('https://firebasestorage.googleapis.com/v0/b/mygroceryapp-80ee5.appspot.com/o/img%2F' + in_img_path + '?alt=media', model)
    print(feature_list)
    print(features)
    indices = recommend(features, feature_list)
    
    for i in range(10):
        row = {}
        row['img_path'] = filenames[indices[0][i]]
        out_img_paths.append(row)
    return jsonify(out_img_paths)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="5000")
    app.run(host="192.168.0.104", port="5000")
