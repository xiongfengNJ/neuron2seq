import tensorflow as tf
import os
import pickle
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
import pandas as pd
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras.layers import Lambda, Flatten, Dense, Multiply, Activation, Conv1D, GRU, GlobalMaxPooling1D, \
    Dropout, Bidirectional, LSTM, Layer, GlobalAveragePooling1D, BatchNormalization, Concatenate, MaxPool1D, Masking, \
    TimeDistributed, RepeatVector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import initializers, Input, Model, constraints, layers, losses, optimizers, \
    Sequential, Input, regularizers
from gensim.models.callbacks import CallbackAny2Vec
import copy
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec

maxlen_word = 20
maxlen_sentence = 300
maxlen = 2000
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select 0 for first GPU or 1 for second
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, text_path):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
        self.text_path = text_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = np.argmax(self.validation_data[1], axis=-1)

        # if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
        #     val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        with open(self.text_path, 'a+') as f:
            f.write(str(_val_f1.item()) + " " + str(_val_recall.item()) + " " + str(_val_precision.item()) + " ")
            f.write("\n")
        return


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, path):
        self.epoch = 0
        self.pre_loss = 0
        self.path_save = path
        self.least_loss = 9999999999
        self.best_model = 0
        if os.path.exists(self.path_save):
            os.remove(self.path_save)
        with open(self.path_save, 'a+') as f:
            f.write("epoch loss\n")

    def on_epoch_end(self, model):
        self.epoch += 1
        cur_loss = model.get_latest_training_loss()
        e_loss = cur_loss - self.pre_loss
        with open(self.path_save, 'a+') as f:
            f.write(str(self.epoch) + " " + str(e_loss) + "\n")
        self.pre_loss = cur_loss
        if e_loss < self.least_loss:
            self.least_loss = e_loss
            self.best_model = copy.deepcopy(model)


def auto_encoder(maxlen, hid_dim, input_dim=256):
    input = Input((maxlen, input_dim,))
    x = Masking(mask_value=0.0)(input)
    x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    # x = Flatten()(x)

    encoder = GRU(hid_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=False,
                  name='encoder')(x)

    # x = BatchNormalization()(encoder)
    x = RepeatVector(maxlen)(encoder)
    x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

    decoder = GRU(input_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

    model = Model(inputs=input, outputs=decoder)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.001, decay=0.001, clipnorm=1.0),
                  # 0.005 # experimental_run_tf_function=False,
                  loss=tf.keras.losses.mean_squared_error)
    return model


def get_TextCNN_model(maxlen, class_num, last_activation='softmax'):
    input = Input(shape=(maxlen, 8,))

    convs = []
    for kernel_size in [2, 3, 5]:
        c = Conv1D(256, kernel_size, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(input)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = K.concatenate((convs))
    x = Sequential([
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.05), use_bias=True),  # 0.01
        layers.Dropout(rate=0.5),
        layers.ReLU(),
        # layers.Dense(16, kernel_regularizer=regularizers.l2(0.01)),  # 0.01
        # layers.Dropout(rate=0.5),
        # layers.ReLU()
    ])(x)

    output = Dense(class_num, activation=last_activation, use_bias=True, kernel_regularizer=regularizers.l2(0.005))(
        x)  # 0.005->0.008
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.001, clipnorm=1., decay=0.001),
                  # 0.005 # experimental_run_tf_function=False,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#
# model = get_TextCNN_model(2000, 5)
# plot_model(model, to_file=r'D:\2020\dl_text_cl\model_textcnn.jpg', show_shapes=True)


def get_TextRNN_model(maxlen, class_num, last_activation='softmax'):
    input = Input((maxlen, 8,))
    x = Masking(mask_value=0.0)(input)
    # x = Bidirectional(GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True))(input)
    x = GRU(128, kernel_regularizer=regularizers.l2(0.01), dropout=0.5, return_sequences=True)(x)
    x = GRU(64, kernel_regularizer=regularizers.l2(0.01), dropout=0.5)(x)
    x = Sequential([
        layers.Dense(32, kernel_regularizer=regularizers.l2(0.1)),
        layers.Dropout(rate=0.5),
        layers.ReLU()])(x)

    output = Dense(class_num, activation=last_activation, kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.003, clipnorm=1., decay=0.0000)  # , experimental_run_tf_function=False
                  , loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#
# # model= get_TextBiRNN_model(2000,9)
# # plot_model(model,to_file=r'D:\2020\dl_text_cl\model_birnn.jpg',show_shapes=True)
#

class Attention_text(layers.Layer):
    def __init__(self, embedding_dim):
        self.w_dense = layers.Dense(embedding_dim, activation='tanh', kernel_regularizer=regularizers.l2(0.1))
        self.u_dense = layers.Dense(embedding_dim, activation='linear', kernel_regularizer=regularizers.l2(0.1))
        super(Attention_text, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w_dense': self.w_dense,
            'u_heads': self.u_dense
        })
        return config

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs):  # input shape [batch_size, seq_length, dim]
        uit = self.w_dense(inputs)  # tanh
        ait = self.u_dense(uit)  # linear
        a = K.exp(ait)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        a = K.squeeze(a, axis=-1)
        weighted_input = inputs * a
        return K.sum(weighted_input, axis=1), a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class simple_att(layers.Layer):
    def __init__(self, embedding_dim, name):
        self.w_dense = tf.keras.layers.Dense(embedding_dim, name=name, activation='softmax',
                                             kernel_regularizer=regularizers.l2(0.001))
        self.supports_masking = True
        super(simple_att, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w_dense': self.w_dense,
        })
        return config

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):  # input shape [batch_size, seq_length, dim]
        QK = self.w_dense(inputs)  # activation = "softmax"
        MV = tf.keras.layers.Multiply()([QK, inputs])

        output = K.sum(MV, axis=1)
        return output, QK

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def get_simple_HAN_model(maxlen_sentence=300, maxlen_word=20, class_num=5, last_activation='softmax',
                         index_test=False, input_dim=3):
    input_word = Input(shape=(maxlen_word, input_dim,))  # [batchsize, word_length, dim(8)]
    x_word = Masking(mask_value=0.0)(input_word)
    em1 = 128
    # x_word = Bidirectional(GRU(em1, return_sequences=True, kernel_regularizer=regularizers.l2(0.0015), dropout=0.5))(
    #     x_word)  # LSTM or GRU

    if not index_test:
        x_word = GRU(em1, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), dropout=0.5)(
            x_word)  # LSTM or GRU
    elif index_test:
        x_word = TimeDistributed(Dense(em1, kernel_regularizer=regularizers.l2(0.001)))(x_word)

    x_word, word_attention_weight = simple_att(em1, 'word_att')(x_word)
    model_word = Model(input_word, x_word)

    # Sentence part
    input = Input(shape=(maxlen_sentence, maxlen_word, input_dim,))  # [batchsize, sentence_length, word_length, dim(8)]
    x_sentence = TimeDistributed(model_word)(input)
    em2 = 128
    # x_sentence = Bidirectional(
    #     GRU(em2, return_sequences=True, kernel_regularizer=regularizers.l2(0.0015), dropout=0.5))(
    #     x_sentence)  # LSTM or GRU

    if not index_test:
        x_sentence = GRU(em2, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), dropout=0.5)(
            x_sentence)  # LSTM or GRU
    elif index_test:
        x_sentence = TimeDistributed(Dense(em2, kernel_regularizer=regularizers.l2(0.001)))(x_sentence)

    x_sentence, sentence_attention_weight = simple_att(em2, 'sentence_att')(x_sentence)

    x_sentence = Sequential([
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.015)),
        layers.Dropout(rate=0.5),
        layers.ReLU()])(x_sentence)

    output = Dense(class_num, activation=last_activation, kernel_regularizer=regularizers.l2(0.01))(x_sentence)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, clipvalue=0.5, decay=0.002),
                  loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
    return model


#
# model = get_HAN_model(300, 20, 5)
# plot_model(model, to_file=r'D:\NMT\dl_text_cl\model_HAN.jpg', show_shapes=True)
#

def get_cnn_gru_model(maxlen_sentence=300, maxlen_word=20, class_num=5, last_activation='softmax'):
    input_word = Input(shape=(maxlen_word, 8,))
    convs = []
    for kernel_size in [2, 3, 5]:
        c = Conv1D(256, kernel_size, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_word)
        c = BatchNormalization()(c, training=False)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x_word = Concatenate()(convs)
    # x_word = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(x_word)

    model_word = Model(inputs=input_word, outputs=x_word)

    # Sentence part
    input = Input(shape=(maxlen_sentence, maxlen_word, 8,))
    x_sentence = TimeDistributed(model_word)(input)
    x_sentence = GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), dropout=0.5)(
        x_sentence)  # LSTM or GRU
    x_sentence, sentence_attention_weight = Attention_text(64)(x_sentence)

    x_sentence = Sequential([
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.05)),
        layers.Dropout(rate=0.5),
        layers.ReLU()]
    )(x_sentence)

    output = Dense(class_num, activation=last_activation, kernel_regularizer=regularizers.l2(0.05))(x_sentence)
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.001, clipnorm=1., decay=0.00),  # experimental_run_tf_function=False,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model =get_cnn_gru_model()a
# plot_model(model, to_file=r'D:\2020\dl_text_cl\model_cnn_gru.jpg', show_shapes=True)


def data_process(data, feature, w2v_path, split_data=False):
    model_w2v = Word2Vec.load(w2v_path)
    wv = model_w2v.wv

    vectors = wv.vectors
    vectors = StandardScaler().fit_transform(vectors)
    # vectors = PCA(3).fit_transform(vectors)

    input_dim = vectors.shape[1]
    print("input_dim", input_dim)

    region_dict = dict(zip(wv.index2word, vectors))

    x_region_list = []
    for i in data.index.tolist():
        temp_words = list(data.loc[i, 'region'].split(' '))
        temp_vector = np.zeros((len(temp_words), input_dim))
        for k, j in enumerate(temp_words):
            try:
                temp_vector[k, :] = region_dict[j]
            except:
                temp_vector[k, :] = region_dict['unknow']
        x_region_list.append(temp_vector)

    x_loc_df = data[['scaled_x', 'scaled_y', 'scaled_z']]
    x_loc_list = []
    for i in x_loc_df.index.tolist():
        temp = np.vstack(x_loc_df.loc[i]).T
        x_loc_list.append(temp)

    x_type_df = data[['node_type']]
    x_type_list = []
    for i in range(len(x_type_df)):
        temp = pd.get_dummies(list(x_type_df.iloc[i, 0]))
        try:
            temp = temp[['g', 'b', 't']]
        except:
            temp = temp[['b', 't']]
        x_type_list.append(temp.values)
    x_region_list_after = []
    x_loc_list_after = []
    x_type_list_after = []
    index_list = data.index.tolist()
    for k1, i in enumerate(index_list):
        temp_nodes = list(data.loc[i, 'node_type'])
        tmp1_region = []
        tmp1_loc = []
        tmp1_type = []
        tmp2_region = []
        tmp2_loc = []
        tmp2_type = []
        for k2, j in enumerate(temp_nodes):
            tmp2_region.append(x_region_list[k1][k2,].astype(float))
            tmp2_loc.append(x_loc_list[k1][k2,].astype(float))
            tmp2_type.append(x_type_list[k1][k2,].astype(float))
            if (j == 't'):
                # if (j == 't') or (temp_nodes[k2 + 1] == 'b'):
                tmp1_region.append(tmp2_region)
                tmp1_loc.append(tmp2_loc)
                tmp1_type.append(tmp2_type)
                # tmp2_* 清空
                tmp2_region = []
                tmp2_loc = []
                tmp2_type = []
        tmp1_region = pad_sequences(tmp1_region, maxlen=maxlen_word, dtype='float32')
        tmp1_type = pad_sequences(tmp1_type, maxlen=maxlen_word, dtype='float32')
        tmp1_loc = pad_sequences(tmp1_loc, maxlen=maxlen_word, dtype='float32')
        x_region_list_after.append(tmp1_region)
        x_type_list_after.append(tmp1_type)
        x_loc_list_after.append(tmp1_loc)
    x_region = pad_sequences(x_region_list_after, maxlen=maxlen_sentence, dtype='float32')
    x_loc = pad_sequences(x_loc_list_after, maxlen=maxlen_sentence, dtype='float32')

    if feature == 2:
        x = np.concatenate([x_loc, x_region], axis=-1)
        input_dim = input_dim + 3

    elif feature == 1:
        x = x_region

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(data['label'])

    class_num = np.unique(y).shape[0]
    y = to_categorical(y, num_classes=class_num)
    # print('number of class:', class_num)

    if split_data:
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(x, y):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
        return x_train, y_train, x_test, y_test, input_dim, class_num
    else:
        return x, y, input_dim, class_num


def data_process_ae(data, w2v_path, maxlen, feature=1):
    model_w2v = Word2Vec.load(w2v_path)
    wv = model_w2v.wv

    vectors = wv.vectors
    vectors = StandardScaler().fit_transform(vectors)

    input_dim = vectors.shape[1]
    print("input_dim", input_dim)

    region_dict = dict(zip(wv.index2word, vectors))

    x_region_list = []
    for i in data.index.tolist():
        temp_words = list(data.loc[i, 'region'].split(' '))
        temp_vector = np.zeros((len(temp_words), input_dim))
        for k, j in enumerate(temp_words):
            try:
                temp_vector[k, :] = region_dict[j]
            except:
                temp_vector[k, :] = region_dict['unknow']
        x_region_list.append(temp_vector)

    x_loc_df = data[['scaled_x', 'scaled_y', 'scaled_z']]
    x_loc_list = []
    for i in x_loc_df.index.tolist():
        temp = np.vstack(x_loc_df.loc[i]).T
        x_loc_list.append(temp)

    x_type_df = data[['node_type']]
    x_type_list = []
    for i in range(len(x_type_df)):
        temp = pd.get_dummies(list(x_type_df.iloc[i, 0]))
        try:
            temp = temp[['g', 'b', 't']]
        except:
            temp = temp[['b', 't']]
        x_type_list.append(temp.values)

    x_region = pad_sequences(x_region_list, maxlen=maxlen, dtype='float32')
    x_loc = pad_sequences(x_loc_list, maxlen=maxlen, dtype='float32')

    if feature == 2:
        x = np.concatenate([x_loc, x_region], axis=-1)
        input_dim = input_dim + 3
    elif feature == 1:
        x = x_region

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(data['label'])

    class_num = np.unique(y).shape[0]
    y = to_categorical(y, num_classes=class_num)

    return x, y, input_dim, class_num
