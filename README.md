# Attribute and Simile Classifiers for Face Verification

## Overview 
1. Low Level Feature extraction: In this part, we extract a vector corresponding to each face region of a face.
2. Attribute Classifier: The extracted vectors for a face is checked for the presence of various attributes like
Male, Attractive, Asian, Indian, etc. and to what extent.
3. Simile Classifier: The extracted vectors for a face is checked for the similarity of face regions with the
reference people.
4. Verification Classifier: Two faces F1 and F2 are passed through all the learnt attribute classifiers and simile
classifiers to obtain a trait vector for each face. Now verification classifier is trained to find whether both these trait vectors belong to the same person’s face or not.

## Datasets used
1. LFW: Used to train the attribute classifier and verification classifier.
2. CelebA: Used to train the attribute classifier.
3. Celebrity Face Recognition Dataset: Used to train the simile classifier.

## Low Level Feature extraction details
1. Face Landmark Detection using the pretrained model shape_predictor_81_face_landmarks.dat
2. Face alignment to tilt faces perfectly parallel to the horizontal axis.
3. Face region extraction
4. Low level feature extracted by calculating edge magnitude and Edge orientation space.
5. For each face region and each space taken into consideration, histogram is calculated over 100 bins.

<p align="center">
  <img width="892" alt="1" src="https://user-images.githubusercontent.com/44245211/137928381-ce172518-4473-4baf-b674-e2711bfe5bfb.png">
  <img width="1086" alt="2" src="https://user-images.githubusercontent.com/44245211/137928422-d4616974-4e76-476c-8761-b0f216edc426.png">
  <img width="683" alt="3" src="https://user-images.githubusercontent.com/44245211/137928415-6a11ca93-00ed-4808-8fde-4130a0a9e46f.png">
</p>

## Attribute Classifier details
1. We loaded the saved histograms.
2. These histograms are divided into train dataset and validation dataset using test size = 0.3
3. We have picked some handful of attributes from the given set of attributes. These attributes are chosen based on the face verification problem.
4. For each chosen attribute, we chose some features using which we have trained our SVM classifiers.
5. While training SVMs, we have performed hyperparameter tuning using grid search.
6. After the model is trained, the models are saved.
7. The models are now tested for accuracy over the validation dataset.

## Simile Classifier details
1. We loaded the saved histograms
2. We generated labels for this dataset. If the histograms belong to the same person, then the label is +1. Else it is -1.
3. The dataset is divided into train data and test data. The train labels -1 is divided using the test size = 0.3 and the train labels +1 is divided using the test size = 0.15
4. We have chosen some handful of reference persons from the dataset.
5. We have trained SVM models for the face regions: eyes, nose and mouth.
6. While training SVMs, we have performed hyperparameter tuning using grid search.
7. After the model is trained, the models are saved.
8. The models are now tested for accuracy over the validation dataset.

## Verification Classifier
1. We divided the dataset randomly into positive and negative images.
2. We extracted the low level features for these pairs of images.
3. We fetched the output of simile and attribute classifiers for these pairs of images (using the low level features extracted above).
4. The output of simile and attribute classifier from each pair of images is concatenated.
5. We generated labels for this dataset. If the outputs of simile and attribute classifiers belong to the same person, then the label is +1. Else it is 0.
6. We trained the SVM.

|True Positive |True Negative|
|-----|--------|
|<img width="486" alt="tp" src="https://user-images.githubusercontent.com/44245211/137930750-22f4708b-5706-4370-ba7d-c78967cada0d.png">|<img width="460" alt="tn" src="https://user-images.githubusercontent.com/44245211/137930691-aa2c9e6a-5a9a-4f6e-b379-f68f14d33c93.png">|

|False Positive |False Negative|
|-----|--------|
|<img width="501" alt="fp" src="https://user-images.githubusercontent.com/44245211/137930733-213b7ff5-f027-4e48-b173-6b6d4b1cdb87.png">  |<img width="502" alt="fn" src="https://user-images.githubusercontent.com/44245211/137930747-8f5cffaa-6020-42d3-a465-5898c015df7b.png">|

## BONUS
Since this paper is from 2009, it focused only upon classification using SVMs. We completed its implementations, and then tried to think of better techniques to improve verification accuracy.
So, we applied various deep learning techniques and in turn, gained a much better understanding of deep learning and how convolutional neural networks help in improving accuracy.

### -- 10% Overall accuracy improvement with Deep Learning -- 

## Directory Structure
- ```src``` folder contains the source code. 
- ```data``` folder contains .npy files which contain calculated histograms. 
