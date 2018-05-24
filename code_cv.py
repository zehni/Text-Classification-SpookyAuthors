import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

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

data = getData('train.csv')

analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()

encoded_data = encodeAuthors(data)

count_vect = CountVectorizer(stop_words='english',
                             token_pattern="\w*[a-z]\w*",
                             max_features=7000,
                             analyzer=stemWords)
tf_matrix = count_vect.fit_transform(encoded_data['text'])

tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)

"""""""""""""""""""""""""""""""""" CV """""""""""""""""""""""""""""""""

#alpha=0.01
#C=7

multi_naive_bayes = MultinomialNB(alpha=0.01)
multi_naive_bayes_scores = cross_val_score(multi_naive_bayes, tfidf_matrix, encoded_data['author'], cv=5, scoring='neg_log_loss')
multi_naive_bayes_loss = np.mean(multi_naive_bayes_scores); multi_naive_bayes_loss
print("\nMultinomial Naive Bayes. Negative Log Loss: %0.3f (+/- %0.2f)" % (-multi_naive_bayes_loss, multi_naive_bayes_scores.std() * 2))

log_reg = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=400, C=7)
log_reg_scores = cross_val_score(log_reg, tfidf_matrix, encoded_data['author'], cv=5, scoring='neg_log_loss')
log_reg_loss = np.mean(log_reg_scores); log_reg_loss 
print("\nLogistic Regression. Negative Log Loss: %0.3f (+/- %0.2f)" % (-log_reg_loss, log_reg_scores.std() * 2))


""""""""""""""""""""""""" Fit Models/Submission """""""""""""""""""""""""

test_data = getData('test.csv')

tf_matrix_test = count_vect.transform(test_data['text'])
tfidf_matrix_test = tfidf_transformer.transform(tf_matrix_test)

# fit models on training set, predict on Kaggle's test set

nbayes = MultinomialNB(alpha=0.01)
nbayes.fit(tfidf_matrix, encoded_data['author'])
nbayes_proba_preds = nbayes.predict_proba(tfidf_matrix_test)

logreg = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=400, C=6)
logreg.fit(tfidf_matrix, encoded_data['author'])
logreg_proba_preds = logreg.predict_proba(tfidf_matrix_test)


# pandas dataframe to write it on csv (for MNB)
nbayes_submission = pd.DataFrame(data=nbayes_proba_preds)
nbayes_submission['id'] = test_data['id']
nbayes_submission.columns = ['EAP', 'HPL', 'MWS', 'id']
nbayes_submission = nbayes_submission[['id', 'EAP', 'HPL', 'MWS']]

# pandas dataframe to write it on csv (for LR)
logreg_submission = pd.DataFrame(data=logreg_proba_preds)
logreg_submission['id'] = test_data['id']
logreg_submission.columns = ['EAP', 'HPL', 'MWS', 'id']
logreg_submission = logreg_submission[['id', 'EAP', 'HPL', 'MWS']]


""""""""""""""""""""""""" Write to CSV """""""""""""""""""""""""""""""""
""""""""""""""""" Be careful not to overwrite this """""""""""""""""""""

# put your won working directory
#nbayes_submission.to_csv("C:/Users/mixal/nbayes_submission3.csv", index=False)
#logreg_submission.to_csv("C:/Users/mixal/logreg_submission3.csv", index=False)























