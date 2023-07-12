import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request

app = Flask(__name__)

umodel = tf.keras.models.load_model('s_model')
model = tf.keras.models.load_model('mfcc_model')
metadata = pd.read_csv("original.csv")

d = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5: 'engine_idling', 6:'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'}
tb = {-1 : 'Label Unknown',0: 'TB Negative', 1: 'TB Positive'}

def urbansoundfunc(filename):
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=np.argmax(umodel.predict(mfccs_scaled_features),axis=1)
    return d[predicted_label[0]]

def func(filename):
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features = np.repeat(mfccs_scaled_features, 1, axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1, 8, 5, 1)
    predicted_label=np.argmax(model.predict(mfccs_scaled_features),axis=1)
    return tb[predicted_label[0]]


def func2(filename):
    ground = -1
    for i in range(0, len(metadata['filename'])):
            if(filename.filename == metadata['filename'][i]):
                 ground = metadata['tb_status'][i]
    return ground

@app.route('/urbansoundpredict',methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return 'No file provided', 400

    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith('.wav'):
        return 'Invalid file type, must be .wav', 400
    preditction = urbansoundfunc(audio_file)
    print(preditction)
    return preditction


@app.route('/predict',methods=['POST'])
def tb_predict():
    if 'audio' not in request.files:
        return 'No file provided', 400

    audio_file = request.files['audio']
    filename = request.files['audio']
    if not audio_file.filename.lower().endswith('.wav'):
        return 'Invalid file type, must be .wav', 400
    preditction = [ func(audio_file), tb[int(func2(filename))] ]
    print(preditction[0])
    return preditction[0]+","+preditction[1]




if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=80)













# import os

# import librosa
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from flask import Flask, jsonify, request

# # from flask_ngrok import run_with_ngrok

# app = Flask(__name__)
# # run_with_ngrok(app)

# model = tf.keras.models.load_model('mfcc_model')

# metadata = pd.read_csv("original.csv")

# d = {-1 : 'Label Unknown',0: 'TB Negative', 1: 'TB Positive'}

# def func(filename):
#     audio, sample_rate = librosa.load(filename)
#     mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
#     mfccs_scaled_features = np.repeat(mfccs_scaled_features, 1, axis=0)
#     mfccs_scaled_features=mfccs_scaled_features.reshape(1, 8, 5, 1)
#     predicted_label=np.argmax(model.predict(mfccs_scaled_features),axis=1)
#     return d[predicted_label[0]]


# def func2(filename):
#     ground = -1
#     for i in range(0, len(metadata['filename'])):
#             if(filename.filename == metadata['filename'][i]):
#                  ground = metadata['tb_status'][i]
#     return ground


# @app.route('/predict',methods=['POST'])
# def predict():
#     if 'audio' not in request.files:
#         return 'No file provided', 400

#     audio_file = request.files['audio']
#     filename = request.files['audio']
#     if not audio_file.filename.lower().endswith('.wav'):
#         return 'Invalid file type, must be .wav', 400
#     preditction = [ func(audio_file), d[int(func2(filename))] ]
#     print(preditction[0])
#     return preditction[0]+","+preditction[1]



# if __name__ == '__main__':
#     app.run(debug = True,host="0.0.0.0",port=80)