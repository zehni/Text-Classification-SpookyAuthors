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
from sklearn import linear_model

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
    x = data['text']
    y = data['author']
    # Splitting data to 70% Train and 30% Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    return x_train, x_test, y_train, y_test

def stemWords(text):
    return (stemmer.stem(w) for w in analyzer(text))

# Count vectorizer for the train data
def countVec(x_train):
    count_vect = CountVectorizer(stop_words='english', 
                                 token_pattern="\w*[a-z]\w*", 
                                 max_features=2000,
                                 analyzer=stemWords)
    x_train_counts = count_vect.fit_transform(x_train)
    
    return x_train_counts

# TF-IDF values based on the count vectorizer
def tfidfTransform(x_train_counts):
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    return x_train_tfidf

# Transform the data to be predicted
def countvecTransform(train_data, data):
    count_vect = CountVectorizer()
    # You need to fit and then transform the test data again
    # To transform the data to be predicted in the same way
    fit = count_vect.fit_transform(train_data)
    transform = []
    # Transform sentence by count vec
    for sentences in data:
        transform.append(count_vect.transform([sentences]))
    return transform

# Multinomial Naive Baysen Classifer
def MultinomialNaiveBaysen(x_train_tfidf, y_train, train_data, data):
    '''
    params:
    x_train_tfidf: TF-IDF values of the train data
    y_train: Labels of the train data
    data: The x of the data to be predicted

    returns:
    prediction
    '''
    model = MultinomialNB().fit(x_train_tfidf, y_train)
    transform = countvecTransform(train_data, data)
    prediction = []
    # Predict the authors of the sentences
    for counts in transform:
        prediction.append(model.predict(counts)[0])
    return prediction

# Linear Suport Vector Machine Classifer
# Penality by default is 1
def LinearSVM(x_train_tfidf, x_train, y_train, x_test, penality=1):
    model = svm.LinearSVC(C=penality)
    model.fit(x_train_tfidf, y_train)
    x_test_transform = countvecTransform(x_train, x_test)
    prediction=[]
    for sentences in x_test_transform:
        prediction.append(model.predict(sentences)[0])
    return prediction

data = getData('train.csv')
# plot = plotInitialData(data)

analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()

encoded_data = encodeAuthors(data)
x_train, x_test, y_train, y_test = splitData(encoded_data)

# Adminstration
count_vectorizer = countVec(x_train)
tfidf_transformation = tfidfTransform(count_vectorizer)

""" Proposed change. """

""" 
Why not just call countVec() for x_test and transform the test data that way?
I don't think it's necessary to call countVecTransform() inside MultinomialNaiveBaysen() function.
"""

# do this instead?
count_vectorizer_test = countVec(x_test)
tfidf_transformation_test = tfidfTransform(count_vectorizer_test)

""" """

print("################ Multinomial Naive Baysen ################ ")
# Prediction on train
t = time.time()
MNB_predict_train = MultinomialNaiveBaysen(tfidf_transformation, y_train, x_train, x_train)
print("Accuracy on the train set ",round(accuracy_score(y_train, MNB_predict_train)*100,2),"%")
print(classification_report(y_train, MNB_predict_train))
print("it took", time.time()-t)
# Prediction on the test
t = time.time()
MNB_predict_test = MultinomialNaiveBaysen(tfidf_transformation, y_train, x_train, x_test)
print("Accuracy on the test set ",round(accuracy_score(y_test, MNB_predict_test)*100,2),"%")
print(classification_report(y_test, MNB_predict_test))
print("it took", time.time()-t)

print("################ Linear Support Vector Machine Classifer ################ ")
print("## Penality = 1")
t = time.time()
LinearSVM_predict = LinearSVM(tfidf_transformation, x_train, y_train, x_test)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("it took", time.time()-t)


print("################ Linear Support Vector Machine Classifer ################ ")
print("## Penality = 0.5")
t = time.time()
LinearSVM_predict = LinearSVM(tfidf_transformation, x_train, y_train, x_test, 0.5)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("it took", time.time()-t)

print("################ Logistic Regression ################ ")
t = time.time()
model = linear_model.LogisticRegression()
model.fit(tfidf_transformation, y_train)
LogisticReg_predict = model.predict(tfidf_transformation_test)
print("Accuracy on the test set LOGISTIC REGRESSION",round(accuracy_score(y_test, LogisticReg_predict)*100,2))
print(classification_report(y_test, LogisticReg_predict))
print("it took", time.time()-t)






