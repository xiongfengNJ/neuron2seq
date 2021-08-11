from init import *
import plotly.graph_objects as go

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_data(neuron_path, neuron_name, neuron_region_path, with_grow='keep', scale_=0, index_itf=False):
    neuron_region = pd.read_csv(neuron_region_path, header=0, index_col=0, sep=',')

    s = neuron(neuron_path, scale_=scale_)
    df = s.neuron2seqs(with_grow=with_grow, index_itf=index_itf)
    df[['x_scaled', 'y_scaled', 'z_scaled']] = StandardScaler().fit_transform(df[['x', 'y', 'z']].values)
    x = df[['x_scaled', 'y_scaled', 'z_scaled', 'node_type', 'region', 'x', 'y', 'z', 'type', 'flag']].values
    # x_scaled y_scaled z_scaled B/T/G region x y z type flag
    try:
        y = neuron_region.loc[neuron_name, 'final_celltype']
    except:
        y = 'unknown'
    return x, y


def generate_dataset(dataset, scale, index_itf):
    print('from ' + dataset + ' swc to seq data')
    data_path = src + '/' + dataset
    if scale == 0:
        data_to_save = src + '/' + dataset + '_data.pickle'
    else:
        data_to_save = src + '/' + str(scale) + '_' + dataset + '_data.pickle'

    print('data saved at ', data_to_save)

    neuron_region_path = src + '/' + '1741_Celltype.csv'
    swc_list = os.listdir(data_path)
    neuron_list = [swc.split('.')[0] for swc in swc_list]
    path_list = [data_path + '/' + swc for swc in swc_list]

    xs = []
    ys = []
    zs = []
    fail_list = []
    for neuron_path, neuron_name in zip(path_list, neuron_list):
        print(neuron_name, " wait....")
        x, y = get_data(neuron_path, neuron_name, neuron_region_path, 'keep', scale_=scale, index_itf=index_itf)
        if y:
            print(neuron_name, " finish")
            xs.append(x)
            ys.append(y)
            zs.append(neuron_name)
        else:
            print(neuron_name, " fails, please check")
            fail_list.append(neuron_name)
            continue

    print("length of generated sequences", len(xs))

    ys = np.array(ys)

    data = pd.DataFrame(
        {'scaled_x': i[:, 0], 'scaled_y': i[:, 1], 'scaled_z': i[:, 2], 'node_type': ''.join(i[:, 3].tolist()),
         'region': ' '.join(
             i[:, 4].tolist()).replace('fiber tracts', 'fiber_tracts'), 'x': i[:, 5], 'y': i[:, 6], 'z': i[:, 7],
         'type': i[:, 8], 'flag': i[:, 9]} for i in xs)
    data['label'] = ys
    data.index = zs
    le = LabelEncoder()
    data['label_'] = le.fit_transform(data['label'])

    fw_data = open(data_to_save, 'wb', 1)
    pickle.dump(data, fw_data)
    fw_data.close()
    print('unknow swc label: \n', fail_list)


# params description:

# scale: Standard Deviation value, to add spatial noises to data, set scale here
# index_itf: unused in study, used to shuffle the sequence order

if __name__ == '__main__':
    src = "../raw_dataset"
    for scale in range(630, 710, 10):  # scale: to add gaussian noise(0 means no noise)
        print("noise sd scale ", scale)
        for dataset in ['1728_picked_test']:  # dataset: choose swc dataset,inputs folder name
            generate_dataset(dataset, scale, index_itf=False)
