import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 


# Read the CSV file, convert it to a dataframe and return the dataframe
# Also returns a success parameter. If the passed fileName is invalid, 
# return failure status. 
def readData(fileName):
    success = False
    dataFrame = pd.DataFrame()

    if fileName.lower().endswith('.csv'):
        try:
            dataFrame = pd.read_csv(fileName)
            success = True
            return dataFrame, success
        except FileNotFoundError:
            print(f"File {fileName} not found. Please check the file path and try again.")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
    else:
        return dataFrame, success
    
# Use readData method to read the a given training .csv file. 
# Load the "content" and "class" column of the dataframe and return
# X as features and Y as responses
def loadTrainingData(fileName):
    # Read the data and store it as a data frame
    dataFrame, _ = readData(fileName)
    dataFrame.drop(["AUTHOR", "DATE", "CONTENT", "VIDEO_NAME"], axis = 1)
    X = dataFrame["CONTENT"]
    Y = dataFrame["CLASS"]

    return X, Y

# Use readData method to read the a given test .csv file.
# Load the "content" column of the dataframe and return
# X as features
def loadTestData(fileName):
    # Read the data and store it as a data frame
    dataFrame, _ = readData(fileName)
    dataFrame.drop(["AUTHOR", "DATE", "CONTENT", "VIDEO_NAME"], axis = 1)
    X = dataFrame["CONTENT"]

    return X, dataFrame

# Fit the model to the a given data set, and return the fitted model
def fitModel(model, X, Y):
    model.fit(X, Y)
    return model

# Predict the observation for a given model and its feature, X
def predictFeatures(model, X):
    Y = model.predict(X)
    return Y

# Compute accuracy between a predicted observation and true observation
# Accuracy score = # of correct predictions / Total # of predictions 
# Accurary = 1 - TRAINING ERROR RATE 
# Y1: True observation
# Y2: Predicted observation
# testType: Type of the test data passed, for example "training" or 
# "test" data
# modelName: Name of the model used, for example "SVM" or "Logistic Regression"
def computeAccuracy(Y1, Y2, testType, modelName):
    score = accuracy_score(Y1, Y2)
    print(f"Accuracy of {testType} using {modelName}: {score}")
    print(f"Classification report of {testType} with {modelName} model:\n {classification_report(Y1, Y2)} ")


#######################################################
# Load training data set and vectorize it
#######################################################

# Read the dataset and create a training set with X and Y
Xtrain, Ytrain = loadTrainingData("../Dataset/train.csv")

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

#######################################################
# Load test data set and vectorize it
#######################################################

comments_test, df_test = loadTestData("../Dataset/test.csv")

# Use TF-IDF vectorizer to vectorize test data and extract features
Xtest_test_features = feature_extraction.transform(comments_test)

#######################################################
# Begin SVM Model
#######################################################

# Create a parameter grid for selecting the best model
# param_grid = {'C': [0.1, 1, 5, 8, 10, 100],  
#               'gamma': [1, 0.5, 0.3, 0.2, 0.1, 0.09, 0.01], 
#               'degree':[0, 1, 2, 3],
#               'kernel': ['rbf', 'linear']}

# Fit the training features to the SVM model. Use the vectorized data
# from TF-IDF vectorizer
# svmModel = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 

# Model parameters chosen from cross-validation.
svmModel = svm.SVC(C=8, degree=0, gamma=0.5)
svmModel = fitModel(svmModel, Xtrain_train_features, Ytrain_train)

# Predict traning and test dataset
predictionTrain_train = predictFeatures(svmModel, Xtrain_train_features)
predictionTrain_test = predictFeatures(svmModel, Xtrain_test_features)
computeAccuracy(Ytrain_train, predictionTrain_train, "training data", "SVM")
computeAccuracy(Ytrain_test, predictionTrain_test, "test data", "SVM")


###### Test SVM Model ########
# Perform prediction on test data set
df_test["CLASS"] = predictFeatures(svmModel, Xtest_test_features)

#Store classified result in a .csv file
df_test.to_csv("../Dataset/output/svmClass.csv", index=False)

#######################################################
# End SVM Model
#######################################################

#######################################################
# Begin Logistic Regression Model
#######################################################

# Create an instance of a logistic regression model
logisticRegModel = fitModel(LogisticRegression(C=4.281332398719396, solver='saga'), Xtrain_train_features, Ytrain_train)

# Predict traning and test dataset
predictionTrain_train = predictFeatures(logisticRegModel, Xtrain_train_features)
predictionTrain_test = predictFeatures(logisticRegModel, Xtrain_test_features)
computeAccuracy(Ytrain_train, predictionTrain_train, "training data", "Logistic Regression")
computeAccuracy(Ytrain_test, predictionTrain_test, "test data", "Logistic Regression")

###### Test Logistic Regression Model ########
# Perform prediction on test data set 
df_test["CLASS"] = predictFeatures(logisticRegModel, Xtest_test_features)

#Store classified result in a .csv file
df_test.to_csv("../Dataset/output/logisticRegressionClass.csv", index=False)

#######################################################
# End Logistic Regression Model
#######################################################