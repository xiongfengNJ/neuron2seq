import matplotlib.pyplot as plt
from model_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select 0 for first GPU or 1 for second


#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             # tf.config.experimental.set_virtual_device_configuration(
#             #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


def data_process_ae(data, w2v_path, feature=1):
    # f = open(data_path, 'rb')
    # data = pickle.load(f)
    # f.close()

    model_w2v = Word2Vec.load(w2v_path)
    wv = model_w2v.wv

    vectors = wv.vectors
    vectors = StandardScaler().fit_transform(vectors)

    input_dim = vectors.shape[1]
    print("input_dim", input_dim)

    region_dict = dict(zip(wv.index2word, vectors))

    x_region_list = []

    data = data.reset_index()
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

    return x, y, input_dim


def plot_learning_curve(history, result_dir, k=None):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.plot(loss, 'b', label='loss')
    ax.plot(val_loss, 'r', label='val_loss')
    ax.legend(['loss', 'val_loss'])
    if k != None:
        plt.savefig(result_dir + '/fig' + '_' + str(k) + '.png', dpi=200)
    else:
        plt.savefig(result_dir + '/fig' + '.png', dpi=200)


def auto_encoder(maxlen, hid_dim, input_dim=256):
    input = Input((maxlen, input_dim,))
    x = Masking(mask_value=0.0)(input)
    x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
    # x = Flatten()(x)

    encoder = GRU(hid_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=False)(x)

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


w2v_path = "../final_data/w2v_model_6dim_from_1200_seu.model"

train_data_path = '../final_data/augment_train_dataset_for_ae.pickle'

batchsz = 64
epochs = 300
maxlen = 2000
hid_dim = 32
cur_time = "test"
result_dir = '../result/auto_encoder_result'
model_to_save = result_dir + '/save_model_' + cur_time

path_txt = model_to_save + '/logs_' + cur_time + '.txt'
if os.path.exists(path_txt):
    os.remove(path_txt)

f = open(train_data_path, 'rb')
train_data = pickle.load(f)
f.close()

data_2 = train_data[(train_data.label != 'unknown') & (train_data.label != 'CP_others')]
# df = pd.DataFrame(data_2['label'].value_counts()).astype(int)
# print('============ before ============')
# for k, v in data_2['label'].value_counts().items():
#     print(k, '=================', v)
# th = 30
# p = df.drop(df[df.label < th].index, axis=0).index.tolist()
# data_2 = data_2[data_2.label.isin(p)]
# print('++++++++++++  after  ++++++++++++++++++')

print(data_2['label'].value_counts())
print(data_2['label'].value_counts())
print('data_2 length: ', len(data_2))
print(len(data_2['label'].value_counts()))

x_train, y_train, input_dim = data_process_ae(data_2, feature=1, w2v_path=w2v_path)
# x_test, y_test, input_dim = data_process_ae(test_data_path, feature=1, w2v_path=w2v_path)

ae = auto_encoder(maxlen, hid_dim=hid_dim, input_dim=input_dim)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_to_save + ".h5",
        monitor='val_loss',
        save_best_only=True,
    )
]
#
history = ae.fit(x_train, x_train, epochs=epochs, batch_size=batchsz, callbacks=callbacks_list,
                 validation_data=(x_train, x_train), shuffle=False)

f_his = open(result_dir + '/history_' + cur_time + '.pickle', "wb")
pickle.dump(history.history, f_his)
f_his.close()
plot_learning_curve(history, result_dir=result_dir)

model_evaluate = auto_encoder(maxlen, hid_dim=hid_dim, input_dim=input_dim)
model_evaluate.load_weights(model_to_save + ".h5")
best_loss = model_evaluate.evaluate(x_train, x_train)
with open(result_dir + '/loss_valacc_' + cur_time + '.csv', 'a+') as f:
    f.write(str(best_loss) + '\n')
