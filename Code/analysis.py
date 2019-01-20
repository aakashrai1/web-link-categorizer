from URLParser import URLParser
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

## Reading the dataset

with open('./data/dataset.txt', 'r') as f:
    content = f.read().split('\n\n')[:-1]

tempData = { 'url': [], 'title': [], 'category': []}
for row in content:
    d = row.split('\n')
    tempData['url'].append(d[0])
    tempData['title'].append(d[1])
    tempData['category'].append(d[2])

dataset = pd.DataFrame(tempData, columns = ["url", "title", "category"])

############################################################################################

## Raw analysis of the data ###

#print("Dataset head")
#print(dataset.head())

print('\n', "Dataset shape")
print(dataset.shape)

print('\n', 'Group info')
print(dataset.groupby('category').size())

import seaborn as sns
sns.countplot(dataset['category'],label="Count")
plt.show()

############################################################################################

## Feature creation using TFIDF vectorizer

corpus = []
for i, row in dataset.iterrows():
    pu = URLParser(row['url'], row['title']).getParsedData()
    tok = []
    tok += pu['tokens'] 
    tok += pu['titleTokens']
    corpus.append(" ".join(tok))

vectorizer = TfidfVectorizer(ngram_range = (1,6) , max_features = 10000)
dtm = vectorizer.fit_transform(corpus)
X = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
y = dataset.iloc[:, 2].values

############################################################################################
    
## Analysis for token length (N-grams)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import chi2, SelectKBest

n_grams_accuracy = []

for r in range(1, 8):
    vectorizer = TfidfVectorizer(ngram_range = (1,r), max_features = 10000)
    vectorizer.fit(corpus)
    dtm = vectorizer.transform(corpus)
    X = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
    y = dataset.iloc[:, 2].values
    cl = SelectKBest(chi2, k = 1000).fit_transform(X, y)
    classifier = RandomForestClassifier(n_estimators=50, max_depth = 50,
                             random_state=0, criterion='gini', oob_score=True)
    scores = cross_val_score(classifier, cl, y, cv = 10, scoring = 'accuracy')
    n_grams_accuracy.append(scores.mean())

plt.plot(range(1, 8), n_grams_accuracy)
plt.xlabel('n-gram word tokens')
plt.ylabel('Cross validation accuracy')
        
############################################################################################

## Analysis for selecting number of features

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

features_accuracy = []

for numF in range(500, 4000, 250):
    cl = SelectKBest(chi2, k = numF).fit_transform(X, y)
    classifier = RandomForestClassifier(n_estimators=50, max_depth = 50,
                             random_state=0, criterion='gini', oob_score=True)
    scores = cross_val_score(classifier, cl, y, cv = 10, scoring = 'accuracy')
    features_accuracy.append(scores.mean())

plt.plot(range(500, 4000, 250), features_accuracy)
plt.xlabel('Number of features')
plt.ylabel('Cross validation accuracy')


############################################################################################
    
## Analysis for selecting number of Principal compenents(PCA) during feature selection process

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

PCA_accuracy = []
LSA_accuracy = []

for numC in range(100, 1500, 100):
    pca = PCA(n_components = numC, svd_solver = 'auto', random_state = 0)
    cl = pca.fit_transform(X)
    classifier = RandomForestClassifier(n_estimators=50, max_depth = 50,
                             random_state=0, criterion='gini', oob_score=True)
    scores = cross_val_score(classifier, cl, y, cv = 10, scoring = 'accuracy')
    PCA_accuracy.append(scores.mean())

for numC in range(100, 1500, 100):
    lsa = TruncatedSVD(n_components = numC, n_iter = 50, random_state = 0)
    cl = lsa.fit_transform(X)
    classifier = RandomForestClassifier(n_estimators=50, max_depth = 50,
                             random_state=0, criterion='gini', oob_score=True)
    scores = cross_val_score(classifier, cl, y, cv = 10, scoring = 'accuracy')
    LSA_accuracy.append(scores.mean())
    
plt.plot(range(100, 1500, 100), PCA_accuracy, 'b', label = 'PCA')
plt.plot(range(100, 1500, 100), LSA_accuracy, 'r', label = 'LSA using SVD')
plt.xlabel('Number of components')
plt.ylabel('Cross validation accuracy')
plt.legend()

############################################################################################

## Various Methods for selecting important features

from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif

# chi2
cl = SelectKBest(chi2, k = 1000).fit_transform(X, y)

# mi - information gain
cl = SelectKBest(mutual_info_classif, k = 1000).fit_transform(X, y)

# variance
from sklearn.feature_selection import VarianceThreshold
cl = VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(X)

# Recursive feature elimination CV
from sklearn.feature_selection import RFECV
from sklearn import svm
estimator = svm.LinearSVC()
cl = RFECV(estimator, step = 1, cv = 10).fit_transform(X, y)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1000, svd_solver = 'auto', random_state = 0)
cl = pca.fit_transform(X)

X = cl

############################################################################################

## Analysis for max Tree depth

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

daccuracy = []

for d in range(30, 100, 10):
    classifier = RandomForestClassifier(n_estimators=50, max_depth = d,
                             random_state=0, criterion='gini', oob_score=True)
    cl = SelectKBest(chi2, k = 1000).fit_transform(X, y)
    scores = cross_val_score(classifier, cl, y, cv = 10, scoring = 'accuracy')
    daccuracy.append(scores.mean())

plt.plot(range(30, 100, 10), daccuracy)
plt.xlabel('Tree Depth')
plt.ylabel('Cross validation accuracy')


from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

daccuracy = []

for d in range(30, 150, 15):
    classifier = RandomForestClassifier(n_estimators=50, max_depth = d,
                             random_state=0, criterion='gini', oob_score=True)
    cl = SelectKBest(chi2, k = 1000).fit_transform(X, y)
    scores = cross_val_score(classifier, cl, y, cv = 10, scoring = 'accuracy')
    daccuracy.append(scores.mean())

plt.plot(range(30, 150, 15), daccuracy)
plt.xlabel('Tree Depth')
plt.ylabel('Cross validation accuracy')


############################################################################################

## Top words selected by Latent Semantic Analysis

from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components = 10, n_iter = 50, random_state = 0)
lsa.fit(X)

terms = vectorizer.get_feature_names()
for i, components in enumerate(lsa.components_):
    wordsInComp = zip(terms, components)
    sortedWords = sorted(wordsInComp, key = lambda x:x[1], reverse = True)[:10]
    print("Concept ", i)
    for term in sortedWords:
        print(term[0])
    print(" ")
    
############################################################################################