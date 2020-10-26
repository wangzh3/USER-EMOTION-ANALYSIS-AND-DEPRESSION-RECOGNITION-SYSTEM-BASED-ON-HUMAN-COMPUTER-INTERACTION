# USER-EMOTION-ANALYSIS-AND-DEPRESSION-RECOGNITION-SYSTEM-BASED-ON-HUMAN-COMPUTER-INTERACTION

## 1. Data source and its organization

The HCI video data are comes from the [AVEC 2014 dataset](https://avec2013-db.sspnet.eu). All data are stored in the AVEC2014_AudioVisual folder. In AVEC2014_AudioVisual/structure subfolder, there is a word file that diecribes the stuucture of data organization. It also provides a template folder system.

## 2. Code
1. module folder: recognize emotion in the images. See eagent. py
2. nn folder: neural network codes. totxt.py is used for generating a txt file which includes all path of data that need to be feeded into neural networks.
3. knn folder: K-nearest neighbor code. emo_dep. py is used to created vectors for traning and testing. knn. py is used for classification. relation. py: evaluate the distance for each pair of average feature vector.
4. toframe. py: convert videos into frames.
5. image2vec1. py: get a vector with length 512 of each images by using a middle layer of VGG19.
6. concentratelabel for predict adv. py: generate a txt file for each video. Each row represents a frame. The column represents the value of Arousal, Dominance and Violence. At the each txt file name, there is a value of depression after the 2nd "_" symbol.
7. concentratelabel512. py: generate a txt file for each video. Each row represents a frame by using a vector of length 512. At the each txt file name, there is a value of depression after the 2nd "_" symbol.