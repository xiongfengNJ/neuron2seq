import neuro_morpho_toolbox
from sklearn.model_selection import StratifiedShuffleSplit
from gensim.models import Word2Vec
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

from model_utils import *


batchsz = 64
maxlen_word = 20
maxlen_sentence = 300

model_to_load = '../final_data/for_robust/save_models/from_1728'

index_test = False
feature = 1
w2v_path = "../final_data/w2v_model_6dim_from_1200_seu.model"
test_datas_path = '../final_data/for_robust/test_data/from_1728/6'
models = os.listdir(model_to_load)
test_datas = os.listdir(test_datas_path)

final_data_df = pd.DataFrame(columns=['delta', 'classifier', 'loss', 'val_acc'])
for test_data in test_datas:
    path_data = test_datas_path + '/' + test_data
    delta = test_data.split('_1728')[0]
    print("standard deviation scale: ", delta)

    f = open(path_data, 'rb')
    test_data_df = pickle.load(f)
    f.close()
    x_test, y_test, input_dim, class_num = data_process(test_data_df, feature=1, w2v_path=w2v_path,
                                                        split_data=False)
    for k, m in enumerate(models):
        print("test model ", k+1)
        classifier = m.split('_')[2].split('.')[0]
        path_m = model_to_load + '/' + m
        tmp_model = get_simple_HAN_model(maxlen_sentence, maxlen_word, class_num, index_test=False, input_dim=input_dim)
        tmp_model.load_weights(path_m)
        tmp_loss, tmp_val_acc = tmp_model.evaluate(x_test, y_test, verbose=0)

        tmp_df = pd.DataFrame(
            {'delta': [delta], 'classifier': [classifier], 'loss': [tmp_loss], 'val_acc': [tmp_val_acc]})
        final_data_df = pd.concat([final_data_df, tmp_df], axis=0)
try:
    final_data_df = final_data_df.astype('float')
except:
    print("astype fail")
    pass

final_data_df.to_csv('final_robust_test_pycharm.csv', sep=',')
# print(final_data_df)

