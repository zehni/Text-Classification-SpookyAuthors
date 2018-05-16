import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# Read the data into a data frame
def getData(path):
    texts = pd.read_csv(path)
    return texts

# Encode the authots 'EAP': 0, 'HPL': 1, 'MWS': 2
def encodeAuthors(data):
    data['author'] = data['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})
    encoded_data = data
    return encoded_data

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

""""""""""""""""""""""""""""""""""""" Evaluation (ROC) """""""""""""""""""""""""""""""""""""

data = getData('train.csv')

analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()

encoded_data = encodeAuthors(data)
tf_matrix = countVec(encoded_data['text'])
tfidf_matrix = tfidfTransform(tf_matrix)

y = label_binarize(encoded_data['author'], classes=[0, 1, 2])

x_train_tf, x_test_tf, y_train, y_test = train_test_split(tf_matrix, encoded_data['author'], test_size=0.30, random_state=42)
x_train_tfidf, x_test_tfidf, _, _ = train_test_split(tfidf_matrix, encoded_data['author'], test_size=0.30, random_state=42)
_, _, y_train_bin, y_test_bin = train_test_split(tfidf_matrix, y, test_size=0.30, random_state=42)

linear_svm = svm.LinearSVC(C=0.5)
y_score = linear_svm.fit(x_train_tfidf, y_train).decision_function(x_test_tfidf)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])





