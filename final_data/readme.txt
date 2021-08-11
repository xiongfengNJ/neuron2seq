405_data.pickle: 405 neuron sequence files
test_data_from_405.pickle & train_data_from_405.pickle: split 405 neuron sequence files into training dataset and test dataset

1728_picked_seu_swc.pickle： contains 1282 neuron sequence files, which is picked from 1728 dataset.
1728_seu_swc_data.pickle: contains 1728 neuorn sequence files.
train_data_from1728 & test_data_from_1728.pickle: split 1282 neuron sequence files into training dataset and test dataset

augment_train_dataset_for_ae.pickle: augment 1282 dataset by adding gaussian noise

ae_model_from_405.h5: trained autoencoder model based on 405_data.pickle
ae_model_from_1728_seu.h5: trained autoencoder model based on 1728_picked_seu_swc.pickle
HAN_model_from_405.h5: trained han model based on 405_data.pickle
HAN_model_from_1728.h5: trained han model based on 1728_picked_seu_swc.pickle


