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

from flask import Flask, request, jsonify
import os
from tqdm import tqdm
import pickle
import keras
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np

# model = keras.models.load_model('./models/resnet_50.h5')

# Get a ResNet50 model
def resnet50_model(classes=1000, *args, **kwargs):
    # Load a model if we have saved one
    if(os.path.isfile('./models/resnet_50.h5') == True):
        return keras.models.load_model('./models/resnet_50.h5')
    # Create an input layer 
    input = keras.layers.Input(shape=(None, None, 3))
    # Create output layers
    output = keras.layers.ZeroPadding2D(padding=3, name='padding_conv1')(input)
    output = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(output)
    output = keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name='bn_conv1')(output)
    output = keras.layers.Activation('relu', name='conv1_relu')(output)
    output = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(output)
    output = conv_block(output, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    output = identity_block(output, 3, [64, 64, 256], stage=2, block='b')
    output = identity_block(output, 3, [64, 64, 256], stage=2, block='c')
    output = conv_block(output, 3, [128, 128, 512], stage=3, block='a')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='b')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='c')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='d')
    output = conv_block(output, 3, [256, 256, 1024], stage=4, block='a')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='b')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='c')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='d')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='e')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='f')
    output = conv_block(output, 3, [512, 512, 2048], stage=5, block='a')
    output = identity_block(output, 3, [512, 512, 2048], stage=5, block='b')
    output = identity_block(output, 3, [512, 512, 2048], stage=5, block='c')
    output = keras.layers.GlobalAveragePooling2D(name='pool5')(output)
    output = keras.layers.Dense(classes, activation='softmax', name='fc1000')(output)
    # Create a model from input layer and output layers
    model = keras.models.Model(inputs=input, outputs=output, *args, **kwargs)
    # Print model
    # print()
    # print(model.summary(), '\n')
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adam(lr=0.01, clipnorm=0.001), metrics=['accuracy'])
    # Return a model
    return model
# Create an identity block
def identity_block(input, kernel_size, filters, stage, block):
    
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create layers
    output = keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(output)
    output = keras.layers.add([output, input])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output
# Create a convolution block
def conv_block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create block layers
    output = keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(output)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input)
    shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
    output = keras.layers.add([output, shortcut])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output
# Train a model
def train():
    # Variables, 25 epochs so far
    epochs = 1
    batch_size = 32
    # train_samples = 10 * 5000 # 10 categories with 5000 images in each category
    # validation_samples = 10 * 1000 # 10 categories with 1000 images in each category
    # img_width, img_height = 32, 32
        # train_samples = 18 # 10 categories with 5000 images in each category
        # validation_samples = 58 # 10 categories with 1000 images in each category
        # img_width, img_height = 1800, 2400
    # Get the model (10 categories)
    model = resnet50_model(10)
    # Create a data generator for training
    # train_data_generator = keras.preprocessing.image.ImageDataGenerator(
    #     rescale=1./255, 
    #     shear_range=0.2, 
    #     zoom_range=0.2, 
    #     horizontal_flip=True)
    # # Create a data generator for validation
    # validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
    #     rescale=1./255,
    #     shear_range=0.2,
    #     zoom_range=0.2, 
    #     horizontal_flip=True)
    # # Create a train generator
    # train_generator = train_data_generator.flow_from_directory( 
    #     './train', 
    #     target_size=(img_width, img_height), 
    #     batch_size=batch_size,
    #     color_mode='rgb',
    #     shuffle=True,
    #     class_mode='categorical')
    # # Create a test generator
    # validation_generator = validation_data_generator.flow_from_directory( 
    #     './test', 
    #     target_size=(img_width, img_height), 
    #     batch_size=batch_size,
    #     color_mode='rgb',
    #     shuffle=True,
    #     class_mode='categorical')
    # # Start training, fit the model
    # model.fit_generator( 
    #     train_generator, 
    #     steps_per_epoch=train_samples // batch_size, 
    #     validation_data=validation_generator, 
    #     validation_steps=validation_samples // batch_size,
    #     epochs=epochs)
    # Save model to disk
    model.save('./models/resnet_50.h5')
    print('Saved model to disk!')

    filenames = []
    for file in os.listdir('img'):
        filenames.append(os.path.join('img', file))
    feature_list = []
    for file in tqdm(filenames):
        feature_list.append(extract_features(file, model))
    pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
    pickle.dump(filenames,open('filenames.pkl','wb'))
    # # Get labels
    # labels = train_generator.class_indices
    # # Invert labels
    # classes = {}
    # for key, value in labels.items():
    #     classes[value] = key.capitalize()
    # # Save classes to file
    # with open('./classes.pkl', 'wb') as file:
    #     pickle.dump(classes, file)
    # print('Saved classes to disk!')

def extract_features(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(expanded_img_array).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def feature_extraction(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(expanded_img_array).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

app = Flask(__name__)

@app.route('/virtualtry')
def virtualtry():
    return "Hello"

@app.route('/', methods=['GET'])
def out_recomms():
    # in_img_path = os.path.join('img', request.args.get('in_img_path'))
    in_img_path = '10000.jpg'
    model_run = request.args.get('model_run')
    # if model_run == 'yes':
    model = resnet50_model(10)
    # model.save('./models/resnet_50.h5')
    # model = keras.models.load_model('./models/resnet_50.h5')
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
    
    for i in range(5):
        row = {}
        row['img_path'] = os.path.basename(filenames[indices[0][i]])
        out_img_paths.append(row)
    return jsonify(out_img_paths)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="5000")
    app.run()
