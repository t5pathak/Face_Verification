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

## BONUS
Since this paper is from 2009, it focused only upon classification using SVMs. We completed its implementations, and then tried to think of better techniques to improve verification accuracy.
So, we applied various deep learning techniques and in turn, gained a much better understanding of deep learning and how convolutional neural networks help in improving accuracy.

## Directory Structure
- ```src``` folder contains the source code. 
- ```data``` folder contains .npy files which contain calculated histograms. 
 
## Running the code
Any changes which need to be made, must be made in the code block under the heading "Main" only

1. Open your software of choice (Google Colab, Jupyter Notebook etc.)
2. Navigate to ```src``` folder. (If using Google Colab, upload the file inside ```src``` from your local PC to the colab repository)
3. Load the file
4. Run the file
5. If you want to make any changes to things like number of robots, goal point of robots, where to save the outputs, whether you want to save outputs or not etc., make necessary changes in the cell block under the heading "Main". The cell block is self-explanatory.
6. Install any necessary libraries if needed
