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
                                 max_features=7000,
                                 analyzer=stemWords)
    tf_matrix = count_vect.fit_transform(text)
    return tf_matrix

# TF-IDF values based on the count vectorizer
def tfidfTransform(matrix):
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(matrix)
    return tfidf_matrix

def ComputeROC(y_labels, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_score[:, i]) # y_labels=y_test_bin
        roc_auc[i] = auc(fpr[i], tpr[i])
    return(fpr, tpr, roc_auc)

def PlotROC(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(fpr, tpr, label='AUC: %.3f' % roc_auc)
    ax.plot([0.045, 0.955], [0.045, 0.955], transform=ax.transAxes, linestyle='dashed', color='tab:red')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize = 18)
    ax.tick_params(labelsize = 14)
    ax.grid(alpha = 0.5)
    ax.legend(fontsize=13)
    plt.show()

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


""""""""""""""""""""""""""""""""""""" SVM Evaluation """""""""""""""""""""""""""""""""""""""

linear_svm = svm.LinearSVC(C=0.5)
y_score_svm = linear_svm.fit(x_train_tfidf, y_train).decision_function(x_test_tfidf)

fpr_svm, tpr_svm, auc_svm = ComputeROC(y_test_bin, y_score_svm)

for author in range(3):
    PlotROC(fpr_svm[author], tpr_svm[author], auc_svm[author])

""""""""""""""""""""""""" Multinomial Naive Bayes Evaluation """""""""""""""""""""""""""

# Multinomia Naive Bayes does not have a decision_function() method,
# so we have to compute the probability estimates for each class with predict_proba()

nbayes = MultinomialNB()
nbayes_model = nbayes.fit(x_train_tfidf, y_train)
nbayes_proba_preds = nbayes_model.predict_proba(x_test_tfidf)
fpr_nbayes, tpr_nbayes, auc_nbayes = ComputeROC(y_test_bin, nbayes_proba_preds)

for author in range(3):
    PlotROC(fpr_nbayes[author], tpr_nbayes[author], auc_nbayes[author])

""""""""""""""""""""""""" Logistic Regression Evaluation """""""""""""""""""""""""""""""

logreg = linear_model.LogisticRegression()
logreg_model = logreg.fit(x_train_tfidf, y_train)
logreg_proba_preds = logreg_model.predict_proba(x_test_tfidf)
fpr_logreg, tpr_logreg, auc_logreg = ComputeROC(y_test_bin, logreg_proba_preds)

for author in range(3):
    PlotROC(fpr_logreg[author], tpr_logreg[author], auc_logreg[author])


""""""""""""""""""""""""""""""""" Plot Comparative Curves """""""""""""""""""""""""""""

for author in range(3):
    if author == 0:
        writer = 'EAP'
    elif author == 1:
        writer = 'HPL'
    else:
        writer = 'MWS'
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(fpr_svm[author], tpr_svm[author], label='SVM AUC: %.3f' % auc_svm[author])
    ax.plot(fpr_nbayes[author], tpr_nbayes[author], label='MNB AUC: %.3f' % auc_nbayes[author])
    ax.plot(fpr_logreg[author], tpr_logreg[author], label='LR AUC: %.3f' % auc_logreg[author])
    ax.set_title('ROC Curve, Author %s' % writer, fontsize = 18)
    ax.tick_params(labelsize = 14)
    ax.grid(alpha = 0.5)
    ax.legend(fontsize=13)
    plt.show()        
    





