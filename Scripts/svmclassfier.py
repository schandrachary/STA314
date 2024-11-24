import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm

from sklearn.model_selection import GridSearchCV

# Below imports are only needed if data cleaning is being performed
# prior to model fitting
# import nltk
# from nltk.corpus import stopwords
# import re
# import spacy
# nlp=spacy.load('en_core_web_sm')

# Load the dataset
df = pd.read_csv("../Dataset/train.csv")

######################################
# Training and predict on training data set
######################################

# Remove irrelevant columns from the dataset
df = df.drop(["DATE", "VIDEO_NAME"], axis = 1)

# Load training features and responses
Xtrain = df["CONTENT"]
Ytrain = df["CLASS"]

# Split the training dataset with 80-20 split
Xtrain_train, Xtrain_test, Ytrain_train, Ytrain_test = \
train_test_split(Xtrain, Ytrain, test_size = 0.2, random_state = 3)

# Change the min_df from 1 to 2, i.e, if a word appears in less than 2 sentences, drop it. 
feature_extraction = TfidfVectorizer(min_df = 2, stop_words = 'english', lowercase = True)
Xtrain_train_features = feature_extraction.fit_transform(Xtrain_train)
Xtrain_test_features = feature_extraction.transform(Xtrain_test)

#let's make sure the labels for Y are in int form e.g 0, 1 and not any other like "0", "1"
Ytrain_train = Ytrain_train.astype('int')
Ytrain_test = Ytrain_test.astype('int')

# Create an instance of a SVM  model
# svmModel = svm.SVC() 

# Create a parameter grid for selecting the best SVM model
param_grid = {'C': [0.1, 1, 5, 8, 10, 100],  
              'gamma': [1, 0.5, 0.3, 0.2, 0.1, 0.09, 0.01], 
              'degree':[0, 1, 2, 3],
              'kernel': ['rbf', 'linear']}

# svmModel = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
# Parameters chosen from cross-validation
svmModel = svm.SVC(C=8, degree=0, gamma=0.5)

# Fit the training features to the SVM model. Use the vectorized data
# from TF-IDF vectorizer
svmModel.fit(Xtrain_train_features, Ytrain_train)

# Fit the training features to the logistic regression model. Use the vectorized data
# from TF-IDF vectorizer
# svmModel.fit(Xtrain_train_features, Ytrain_train) 

# Predict traning and test dataset
predictionTrain_train = svmModel.predict(Xtrain_train_features)
predictionTrain_test = svmModel.predict(Xtrain_test_features)

# Accuracy score = # of correct predictions / Total # of predictions 
# Accurary = 1 - TRAINING ERROR RATE 
accurayTrain_train = accuracy_score(Ytrain_train, predictionTrain_train)
print(f"Accuracy of traning data using SVM: {accurayTrain_train}")
accuracyTrain_test = accuracy_score(Ytrain_test, predictionTrain_test)
print(f"Accuracy of test data using SVM: {accuracyTrain_test}")


######################################
# Traning and predict on test data set
######################################

# Read the test data set and extract comments column
df_test = pd.read_csv("../Dataset/test.csv")
comments_test = df_test["CONTENT"]

# Use TF-IDF vectorizer to vectorize test data and extract features
Xtest_test_features = feature_extraction.transform(comments_test)

# Perform prediction on test data set
predictionTest_test = svmModel.predict(Xtest_test_features) 
df_test["CLASS"] = predictionTest_test

# Drop every column except for commentID and Class
df_test = df_test.drop(["AUTHOR", "DATE", "CONTENT", "VIDEO_NAME"], axis = 1)

#Store classified result in a .csv file
df_test.to_csv("../Dataset/output/svmClass.csv", index=False)
