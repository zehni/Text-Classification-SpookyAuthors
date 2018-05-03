'''
author: Zehni Khairullah
May 2 2018
'''

import nltk
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm


# Read the data into a data frame
def getData(path):
    texts = pd.read_csv(path)
    return texts

# Group sentences by the authors
def groupbyAuthor(data):
    byAuthor = data.groupby('author')
    return byAuthor

# Get the unique authors names as a list
def getUniqueAuthors(data):
    authors = list(set(data['author']))
    return authors

# Plot initial data as bar graph
def plotInitialData(data):
    authors = getUniqueAuthors(data)
    byAuthor = groupbyAuthor(data)
    plt.figure()
    plt.bar(authors, byAuthor.count()['text'])
    plt.show()

# Plot initial data as bar graph. (Now it's working)
def plotInitialData2(data):
    data.author.value_counts().plot(kind='bar', rot=0)
    plt.show()

# Encode the authots 'EAP': 0, 'HPL': 1, 'MWS': 2
def encodeAuthors(data):
    data['author'] = data['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})
    encoded_data = data
    return encoded_data

# Split the data to a train set and a test set
def splitData(data):
    X = data['text']
    y = data['author']
    # Splitting data to 70% Train and 30% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test

# Count vectorizer for the train data
def countVec(X_train):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    return X_train_counts

# TF-IDF values based on the count vectorizer
def tfidfTransform(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf

# Transform the data to be predicted
def countvecTransform(train_data, data):
    count_vect = CountVectorizer()
    # You need to fit and the transform the test data again
    # To transform the data to be predicted in the same way
    fit = count_vect.fit_transform(train_data)
    transform = []
    # Transform sentence by count vec
    for sentences in data:
        transform.append(count_vect.transform([sentences]))
    return transform
# Multinomial Naive Baysen Classifer
def MultinomialNaiveBaysen(X_train_tfidf, y_train, train_data, data):
    '''
    params:
    X_train_tfidf: TF-IDF values of the train data
    y_train: Labels of the train data
    data: The X of the data to be predicted

    returns:
    prediction
    '''
    model = MultinomialNB().fit(X_train_tfidf, y_train)
    transform = countvecTransform(train_data, data)
    prediction = []
    # Predict the authors of the sentences
    for counts in transform:
        prediction.append(model.predict(counts)[0])
    return prediction

# Linear Suport Vector Machine Classifer
# Penality by default is 1
def LinearSVM(X_train_tfidf, X_train, y_train, X_test, penality=1):
    model = svm.LinearSVC(C=penality)
    model.fit(X_train_tfidf, y_train)
    X_test_transform = countvecTransform(X_train, X_test)
    prediction=[]
    for sentences in X_test_transform:
        prediction.append(model.predict(sentences)[0])
    return prediction


data = getData('train.csv')
# plot = plotInitialData(data)

encoded_data = encodeAuthors(data)
X_train, X_test, y_train, y_test = splitData(encoded_data)

# Adminstration
count_vectorizer = countVec(X_train)
tfidf_transformation = tfidfTransform(count_vectorizer)


print("################ Multinomial Naive Baysen ################ ")
# Prediction on train
t = time.time()
MNB_predict_train = MultinomialNaiveBaysen(tfidf_transformation, y_train, X_train, X_train)
print("Accuracy on the train set ",round(accuracy_score(y_train, MNB_predict_train)*100,2),"%")
print(classification_report(y_train, MNB_predict_train))
print("it took", time.time()-t)
# Prediction on the test
t = time.time()
MNB_predict_test = MultinomialNaiveBaysen(tfidf_transformation, y_train, X_train, X_test)
print("Accuracy on the test set ",round(accuracy_score(y_test, MNB_predict_test)*100,2),"%")
print(classification_report(y_test, MNB_predict_test))
print("it took", time.time()-t)

print("################ Linear Support Vector Machine Classifer ################ ")
print("## Penality = 1")
t = time.time()
LinearSVM_predict = LinearSVM(tfidf_transformation, X_train, y_train, X_test)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("it took", time.time()-t)


print("################ Linear Support Vector Machine Classifer ################ ")
print("## Penality = 0.5")
t = time.time()
LinearSVM_predict = LinearSVM(tfidf_transformation, X_train, y_train, X_test, 0.5)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("it took", time.time()-t)
