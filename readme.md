### Notes:

#### Contents in this repository : 

1. in 'src' folder :

   auto_encoder.py: to train autoencoder model

   classifier.py: to train HAN model

   dataset_gen.py: to transform swc files to sequences

   model_utils.py: deep learning models are defined here

   parallel_persistence_homology.py: to evaluate persistent homology based framework [Li Y, D Wang, Ascoli G A, et al. Metrics for comparing neuronal tree shapes based on persistent homology[J]. Plos One, 2016, 12(8): e0182184] on our dataset.

   robust_test.py: to test the robustness of HAN model.
   
2.  in 'jupyter' folder: to generate figures in our paper.

3.  in 'raw_dataset' folder: SWC files used in this study.

4.  in 'result' folder: training results of the three models.



#### Large file
Before you run the codes, you need place several files in folder "final_data", please read this part and download necessary files.

Due to the github policy, large file exceeding 100M can not be submitted. So we provide these files in our local server, please go to http://101.43.104.173:8500/, download files you may need, and place them in directories as shown in figure below.

See figure below:

![image-20210810222026981](./readme.assets/image-20210810222026981.png)
