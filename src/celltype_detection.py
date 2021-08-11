import sys

sys.path.append("../src/")
import os
import shutil
from model_utils import *
from init import *
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

sys.setrecursionlimit(1000000)
import numpy as np

from keras.callbacks import LambdaCallback

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def auto_encoder(maxlen, hid_dim, input_dim=256):
    input = Input((maxlen, input_dim,))
    x = Masking(mask_value=0.0)(input)
    x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

    encoder = GRU(hid_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=False,
                  name='encoder')(x)

    x = RepeatVector(maxlen)(encoder)
    x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

    decoder = GRU(input_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

    model = Model(inputs=input, outputs=decoder)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.001, decay=0.001, clipnorm=1.0), loss=tf.keras.losses.mean_squared_error)
    return model


maxlen = 2000
hid_dim = 32

base = '../final_data/'
w2v_path = base + "w2v_model_6dim_1200plus.model"

input_dim = 6
data_path = base + "405_data.pickle"
f = open(data_path, 'rb')
data = pickle.load(f)
f.close()
# reconstruction swc from seu
data_path_seu = '../final_data/ae_seu.pickle'

f = open(data_path_seu, 'rb')
data_seu = pickle.load(f)
f.close()

if __name__ == "__main__":
    final_result = []
    for i in range(27):
        model_ae = r'D:\2020\dl_text_cl\auto_encoder\30\\' + str(i + 1) + '\\save_model_test.h5'
        ae = auto_encoder(maxlen, hid_dim=hid_dim, input_dim=input_dim)
        ae.load_weights(model_ae)

        x, y, input_dim, class_num = data_process_ae(data_path, w2v_path, feature=1)
        encoder = tf.keras.models.Model(inputs=ae.input, outputs=ae.get_layer('encoder').output)

        x_class, y_class, input_dim, class_num = data_process(data_path, feature=1, w2v_path=w2v_path,
                                                              split_data=False)

        model = get_simple_HAN_model(maxlen_sentence, maxlen_word, class_num=5, index_test=False, input_dim=input_dim)
        model.load_weights(base + 'han_model.h5')

        x_vector = encoder.predict(x)
        # print("x_vector shape: ", x_vector.shape)
        x_vector = StandardScaler().fit_transform(x_vector)
        # print("StandardScalized x_vector: ", x_vector.shape)

        # ae_pred: autoencoder output
        ae_pred = ae.predict(x.copy())

        # x_seu_class,y_seu_class : used in HAN
        x_seu_class, y_seu_class, input_dim_new, class_num = data_process(data_path_seu, feature=1, w2v_path=w2v_path,
                                                                          split_data=False)
        # x_seu, y_seu: used in autoencoder
        x_seu, y_seu, input_dim_new, class_num = data_process_ae(data_path_seu, w2v_path, feature=1)

        # HAN model tags swcs with 5 labels
        y_seu_class_pred = model.predict(x_seu_class)

        # x_vector_seu: encoder output
        x_vector_seu = encoder.predict(x_seu)
        x_vector_seu = StandardScaler().fit_transform(x_vector_seu)
        # print("StandardScalized x_vector_new: ", x_vector_seu.shape)

        mse = tf.keras.losses.MeanSquaredError()

        # ae_pred_seu: autoencoder outputu
        ae_pred_seu = ae.predict(x_seu.copy())
        # print("ae_pred_seu shape: ", ae_pred_seu.shape)

        le = preprocessing.LabelEncoder()
        le.fit(data['label'])
        region_num_dict = {-1: 'unknow', 0: 'CP', 1: 'LGd', 2: 'MG', 3: 'SSp-L5', 4: 'VPM'}
        # print("label encoder received classes: ", le.classes_)

        data_seu['pred_num'] = np.argmax(y_seu_class_pred, axis=1)
        data_seu['trained_type'] = -1
        data_seu.loc[data_seu['label'].isin(['CP', 'MG', 'LGd', 'VPM', 'SSp-L5']), 'trained_type'] = 1

        # record the loss and score for each swc
        new_scores = []

        for i in range(len(ae_pred_seu)):
            loss = mse(x_seu[i, :, :], ae_pred_seu[i, :, :]).numpy()

            # transform the loss to score;
            # higher score means swc is more likely to be classified to knonwn type.
            new_scores.append(1 / abs(loss))

        new_scores = np.array(new_scores)
        new_scores = new_scores.reshape(len(new_scores), 1)

        # find the best threshold for the classifier
        best_acc = 0
        best_th = -100
        for th in new_scores:
            new_label_score = np.zeros((120, 1)) - 1
            new_label_score[new_scores >= th] = 1
            cur_acc = metrics.accuracy_score(data_seu['trained_type'], new_label_score)
            if cur_acc >= best_acc:
                best_acc = cur_acc
                best_th = th
        # print('best_acc', best_acc)
        # print('best thresh, ', best_th)

        new_label_score = np.zeros((120, 1)) - 1
        new_label_score[new_scores >= best_th] = 1

        final_result.append(metrics.accuracy_score(data_seu['trained_type'], new_label_score))

    print(final_result)