import pickle
import sys

import matplotlib.pyplot as plt

from model_utils import *

if len(sys.argv) == 1:
    cur_time = 'test'
else:
    cur_time = sys.argv[1]


def plot_learning_curve(history, k=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(acc, 'b', label='acc')
    ax[0].plot(val_acc, 'r', label='val_acc')
    ax[0].legend(['acc', 'val_acc'])

    ax[1].plot(loss, 'b', label='loss')
    ax[1].plot(val_loss, 'r', label='val_loss')
    ax[1].legend(['loss', 'val_loss'])
    if k:
        plt.savefig('../result/classifier_result/fig' + '_' + str(k) + '.png', dpi=300)
    else:
        plt.savefig('../result/classifier_result/fig' + '.png', dpi=300)


# HAN划分句子的方法为以一个t去划分句子
if __name__ == '__main__':
    batchsz = 64
    maxlen_word = 20
    maxlen_sentence = 300
    epochs = 400

    model_to_save = '../result/classifier_result/save_model_' + cur_time

    path_txt = '../result/classifier_result/logs_' + cur_time + '.txt'
    if os.path.exists(path_txt):
        os.remove(path_txt)

    w2v_path = "../final_data/w2v_model_6dim_from_1200_seu.model"
    try:
        tmp = w2v_path.index('1700')
        print('w2v data from 1700')
    except:
        tmp = w2v_path.index('1200')
        print('w2v data from 1200')

    feature = 1  # feature: 1--consider region only; 2--consider region and xyz
    run_type = 'normal'  # run_type: normal or loc ;

    if run_type == 'loc':
        print("for robust test")
        print("split train/test dataset beforehand")
        loc_path = '../final_data/'
        train_data_path = loc_path + 'train_data_from_1728.pickle'
        test_data_path = loc_path + 'test_data_from_1728.pickle'

        f = open(train_data_path, 'rb')
        train_data = pickle.load(f)
        f.close()

        f = open(test_data_path, 'rb')
        test_data = pickle.load(f)
        f.close()

        print("train value counts: \n", train_data['label'].value_counts())
        print("test value counts: \n", test_data['label'].value_counts())

        print("train shape: ", train_data.shape)
        print("test shape: ", test_data.shape)

        x_train, y_train, input_dim, class_num = data_process(train_data, feature=feature, w2v_path=w2v_path,
                                                              split_data=False)
        x_test, y_test, input_dim, class_num = data_process(test_data, feature=feature, w2v_path=w2v_path,
                                                            split_data=False)
        print("class num: ", class_num)

    elif run_type == 'normal':
        print('run_type is normal')
        data_path = '../final_data/1728_picked_seu_swc.pickle'
        f = open(data_path, 'rb')
        data = pickle.load(f)
        f.close()
        data_2 = data[(data.label != 'unknown') & (data.label != 'CP_others')]
        df = pd.DataFrame(data_2['label'].value_counts()).astype(int)
        print('============ before ============')
        for k, v in data_2['label'].value_counts().items():
            print(k, '=================', v)

        th = 30
        p = df.drop(df[df.label < th].index, axis=0).index.tolist()
        data_2 = data_2[data_2.label.isin(p)]
        print('++++++++++++  after  ++++++++++++++++++')
        print(data_2['label'].value_counts())

        x_train, y_train, x_test, y_test, input_dim, class_num = data_process(data_2, feature=feature,
                                                                              w2v_path=w2v_path, split_data=True)


        print('x_train shape: ', x_train.shape)
        print('y_train shape: ', y_train.shape)
        print('x_test shape: ', x_test.shape)
        print('y_test shape: ', y_test.shape)

    print("data prepared")
    print("class number: ", class_num)
    model = get_simple_HAN_model(maxlen_sentence, maxlen_word, class_num, index_test=False, input_dim=input_dim)
    model_evaluate = get_simple_HAN_model(maxlen_sentence, maxlen_word, class_num, index_test=False,
                                          input_dim=input_dim)
    model.summary()

    print("model initialized")

    #
    # log_dir = r'D:\2020\dl_text_cl\tensorboard\logs'
    # # print(log_dir)
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [
        # This callback will interrupt training when we have stopped improving
        # keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=50,
        # ),
        # This callback will save the current weights after every epoch
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_to_save + ".h5",
            monitor='val_accuracy',
            save_best_only=True,
        ),

        Metrics((x_test, y_test), path_txt),

        # keras.callbacks.ReduceLROnPlateau(
        #     # This callback will monitor the validation loss of the model
        #     monitor='val_accuracy',
        #     factor=0.1,
        #     patience=15
        # ),

        # LambdaCallback(
        #     on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
        #                                               logs['loss'])
        # ),

        # 定义TensorBoard对象.histogram_freq 如果设置为0，则不会计算直方图。
        # tensorboard_callback
    ]
    #
    print("training starts here")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batchsz, callbacks=callbacks_list,
                        validation_data=(x_test, y_test), shuffle=False)

    f_his = open('../result/classifier_result/history_' + cur_time + '.pickle', "wb")
    pickle.dump(history.history, f_his)
    f_his.close()
    plot_learning_curve(history, k=cur_time)
    model_evaluate.load_weights(model_to_save + ".h5")
    best_loss, best_val_acc = model_evaluate.evaluate(x_test, y_test)
    with open('../result/classifier_result/loss_valacc_' + cur_time + '.csv', 'a+') as f:
        f.write(str(best_loss) + ' ' + str(best_val_acc) + '\n')
