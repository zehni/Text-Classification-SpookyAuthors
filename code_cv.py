import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

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


data = getData('train.csv')

analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()

encoded_data = encodeAuthors(data)
tf_matrix = countVec(encoded_data['text'])
tfidf_matrix = tfidfTransform(tf_matrix)

"""""""""""""""""""""""""""""""""" CV """""""""""""""""""""""""""""""""

multi_naive_bayes = MultinomialNB()
multi_naive_bayes_scores = cross_val_score(multi_naive_bayes, tfidf_matrix, encoded_data['author'], cv=5)
multi_naive_bayes_est_error = np.mean(multi_naive_bayes_scores); multi_naive_bayes_est_error
print("\nMultinomial Naive Bayes. Accuracy: %0.3f (+/- %0.2f)" % (multi_naive_bayes_est_error, multi_naive_bayes_scores.std() * 2))

linear_svm = svm.LinearSVC(C=0.5)
linear_svm_scores = cross_val_score(linear_svm, tfidf_matrix, encoded_data['author'], cv=5)
linear_svm_est_error = np.mean(linear_svm_scores); linear_svm_est_error
print("\nLinear SVM. Accuracy: %0.3f (+/- %0.2f)" % (linear_svm_est_error, linear_svm_scores.std() * 2))

log_reg = linear_model.LogisticRegression()
log_reg_scores = cross_val_score(log_reg, tfidf_matrix, encoded_data['author'], cv=5)
log_reg_est_error = np.mean(log_reg_scores); log_reg_est_error
print("\nLogistic Regression. Accuracy: %0.3f (+/- %0.2f)" % (log_reg_est_error, log_reg_scores.std() * 2))
































