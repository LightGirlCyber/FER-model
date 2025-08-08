Facial-Emotion-Recognition Model (FER) - computer vision
--------------------------------------------------------

Timeline:
---------
Jun 2025 - Aug 2025Jun 2025 - Aug 2025

Objective:
----------
The main objective was to develop an FER model to detect 9 emotions, 7 of which were included in the popularly used dataset FER 2013 , them being:
Happy ,angry, sad, neutral, disgusted, surprised, and fear. In addition , I've added 2 other labels being kissing/duck-face and sleepy as a challenge to expand on the regularly classified emotions in most FER models.

Pipeline Framework:
-------------------
1. Gathering and filtering a sufficient set of images for the additional labels (sleepy and kissing).
2. Data Preparation & Preprocessing via image data augmentation, oversampling, and adding class weights.
3. Creating 5 block-deep custom CNN (rather than using a prebuilt model to build a lighter model that wouldn't be heavy to load if the model were to be implemented in a future application).
4.Model compilation and training:
For optimal performance of this model, Adam optimizer, loss function and Callbacks were utilized . 

5.Evaluation
------------
Model was trained and evaluated using metrics such as accuracy, precision, recall and f1-score on both train and test data. Additionally , plotting a confusion matrix aided in identifying which label the model was identifying well and which labels the model confused.

Used libraries:
--------------
This model was built using TensorFlow/Keras, visualized using pandas, matplotlib, seaborn and evaluated using Scikit-learn and tested using opencv

Contributers:
-------------
Nour Mohamed Elsisi

References:
------------
Oversampling approach inspired by charlesanjah on Kaggle .
Note: code was adapted and restructured 
Original: https://www.kaggle.com/code/charlesanjah/optimising-facial-emotion-detection-efficientnet

LICENSE
-------
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](LICENSE) file for details.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)