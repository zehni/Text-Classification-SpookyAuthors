'''
author: Zehni Khairullah
May 2 2018
'''

import nltk
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble

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

# Plot initial data as bar graph.
def plotInitialData(data):
    data.author.value_counts().plot(kind='bar', rot=0)
    plt.show()

# Encode the authots 'EAP': 0, 'HPL': 1, 'MWS': 2
def encodeAuthors(data):
    data['author'] = data['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})
    encoded_data = data
    return encoded_data

# Split the data to a train set and a test set
def splitData(x, labels):
    # Splitting data to 70% Train and 30% Test
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.30, random_state=42)
    return x_train, x_test, y_train, y_test

def stemWords(text):
    return (stemmer.stem(w) for w in analyzer(text))

# Count vectorizer for the whole data set.
def countVec(text):
    count_vect = CountVectorizer(stop_words='english',
                                 token_pattern="\w*[a-z]\w*",
                                 max_features=2000,
                                 analyzer=stemWords)
    tf_matrix = count_vect.fit_transform(text)
    return tf_matrix

# TF-IDF values based on the count vectorizer
def tfidfTransform(matrix):
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(matrix)
    return tfidf_matrix

# Multinomial Naive Baysen Classifer
def MultinomialNaiveBaysen(x_train, y_train, x_test):
    '''
    params:
    x_train: Matrix of TF or TFIDF values for training set.
    y_train: Labels to learn.
    x_test: Evaluation on the test set.

    returns:
    Predictions on the test set.
    '''
    model = MultinomialNB().fit(x_train, y_train)
    # Predict the authors of the sentences
    prediction = model.predict(x_test)
    return prediction

# Linear Suport Vector Machine Classifer
# Penality by default is 1
def LinearSVM(x_train, y_train, x_test, penalty=1):
    '''
    params:
    x_train: Matrix of TF or TFIDF values for training set.
    y_train: Labels to learn.
    x_test: Evaluation on the test set.
    penalty: Penalty (C) value.
    
    returns:
    Predictions on the test set.
    '''
    model = svm.LinearSVC(C=penalty)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction

def LogisticReg(x_train, y_train, x_test):
    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction

def RandomForest(x_train, y_train, x_test, 
                 n_estimators, 
                 criterion, 
                 max_features, 
                 max_depth, 
                 n_jobs):
    """
    params:
    See http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
    returns:
    Predictions on the test set.
    """
    model = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                            criterion=criterion,
                                            max_features=max_features,
                                            max_depth=max_depth,
                                            n_jobs=n_jobs)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return(prediction)            

data = getData('train.csv')
plot = plotInitialData(data)

analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()

encoded_data = encodeAuthors(data)
tf_matrix = countVec(encoded_data['text'])
tfidf_matrix = tfidfTransform(tf_matrix)

x_train_tf, x_test_tf, y_train, y_test = splitData(tf_matrix, encoded_data['author'])
x_train_tfidf, x_test_tfidf, _, _= splitData(tfidf_matrix, encoded_data['author'])

print("\n################ Multinomial Naive Baysen ################ ")
# Prediction on train
t = time.time()
MNB_predict_train = MultinomialNaiveBaysen(x_train_tfidf, y_train, x_train_tfidf)
print("Accuracy on the train set ",round(accuracy_score(y_train, MNB_predict_train)*100,2),"%")
print(classification_report(y_train, MNB_predict_train))
print("It took", round(time.time()-t, 2), "seconds.")

# Prediction on the test
t = time.time()
MNB_predict_test = MultinomialNaiveBaysen(x_train_tfidf, y_train, x_test_tfidf)
print("Accuracy on the test set",round(accuracy_score(y_test, MNB_predict_test)*100,2),"%")
print(classification_report(y_test, MNB_predict_test))
print("It took", round(time.time()-t, 2), "seconds.")

print("\n################ Linear Support Vector Machine Classifer ################ ")
print("## Penalty = 1")
t = time.time()
LinearSVM_predict = LinearSVM(x_train_tfidf, y_train, x_test_tfidf, penalty=1)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2), "%")
print(classification_report(y_test, LinearSVM_predict))
print("It took", round(time.time()-t, 2), "seconds.")

print("\n################ Linear Support Vector Machine Classifer ################ ")
print("## Penalty = 0.5")
t = time.time()
LinearSVM_predict = LinearSVM(x_train_tfidf, y_train, x_test_tfidf, penalty=0.5)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("It took", round(time.time()-t, 2), "seconds.")

print("\n##################### Logistic Regression ####################### ")
t = time.time()
LogReg_predict = LogisticReg(x_train_tfidf, y_train, x_test_tfidf)
print("Accuracy on the test set Logistic Regression",round(accuracy_score(y_test, LogReg_predict)*100,2))
print(classification_report(y_test, LogReg_predict))
print("It took", round(time.time()-t, 2), "seconds.")

print("\n##################### Random Forest ####################### ")
t = time.time()
RF_predict = RandomForest(x_train_tfidf, y_train, x_test_tfidf,
                          n_estimators=500, 
                          criterion='entropy',
                          max_features='sqrt',
                          max_depth=500,
                          n_jobs=1)
print("Accuracy on the test set Random Forest",round(accuracy_score(y_test, RF_predict)*100,2))
print(classification_report(y_test, RF_predict))
print("It took", round(time.time()-t, 2), "seconds.")
      













