import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

from URLParser import URLParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from sklearn.feature_selection import chi2, SelectKBest
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score


# Importing the dataset
with open('./data/dataset.txt', 'r') as f:
    content = f.read().split('\n\n')[:-1]

tempData = { 'url': [], 'title': [], 'category': []}
for row in content:
    d = row.split('\n')
    tempData['url'].append(d[0])
    tempData['title'].append(d[1])
    tempData['category'].append(d[2])

dataset = pd.DataFrame(tempData, columns = ["url", "title", "category"])


# Raw analysis
print('\n', 'Group info')
print(dataset.groupby('category').size())
# Plot showing number of data entries in each category
sns.countplot(dataset['category'],label="Count")
plt.show()


# Creating corpus of words
corpus = []
for i, row in dataset.iterrows():
    pu = URLParser(row['url'], row['title']).getParsedData()
    tok = pu['tokens'] + pu['titleTokens']
    corpus.append(" ".join(tok))
    
    
# Vectorizing the dataset
vectorizer = TfidfVectorizer(ngram_range=(1,5), max_features=10000)
dtm = vectorizer.fit_transform(corpus)
X = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names())
y = dataset.iloc[:, 2].values


# Encoding target variable
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)


# Selecting top features using chi2
sel = SelectKBest(chi2, k=2200).fit(X, y)
X = sel.transform(X)


# Generate a wordcloud from top features
mask = sel.get_support()
topFeatures = []

for bool, feature in zip(mask, vectorizer.get_feature_names()):
    if bool:
        topFeatures.append(feature)

topWords = " ".join(topFeatures)
wordcloud = WordCloud(width=1600, height=800).generate(topWords)
plt.figure()
plt.subplots(figsize=(16,9))
wordcloud = WordCloud(background_color="white", max_font_size=30, max_words=120, relative_scaling=0.3).generate(topWords)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#wordcloud.to_file("wordcloud.png")


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1994)


def evaluate(y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n%s" % cm)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))


# Decision tree
dtClassifier = DecisionTreeClassifier(random_state=1994, max_depth=60)
startTime = time.time()
dtClassifier.fit(X_train, y_train)
print("Decision Tree training time: %s" % (time.time() - startTime))
startTime = time.time()
y_pred_dt = dtClassifier.predict(X_test)
print("Decision Tree prediction time: %s" % (time.time() - startTime))
evaluate(y_pred_dt)


# Naive bayes
nbClassifier = MultinomialNB()
startTime = time.time()
nbClassifier.fit(X_train, y_train)
print("Naive Bayes training time: %s" % (time.time() - startTime))
startTime = time.time()
y_pred_nb = nbClassifier.predict(X_test)
print("Naive Bayes prediction time: %s" % (time.time() - startTime))
evaluate(y_pred_nb)


# Random Forest
rfClassifier = RandomForestClassifier(n_estimators=50, max_depth=60,
                             random_state=1994, criterion='gini', oob_score=True)
startTime = time.time()
rfClassifier.fit(X_train, y_train)
print("Random forest training time: %s" % (time.time() - startTime))
startTime = time.time()
y_pred_rf = rfClassifier.predict(X_test)
print("Random forest prediction time: %s" % (time.time() - startTime))
evaluate(y_pred_rf)


# KNN
knnClassifier = KNeighborsClassifier(n_neighbors=50)
startTime = time.time()
knnClassifier.fit(X_train, y_train)
print("KNN training time: %s" % (time.time() - startTime))
startTime = time.time()
y_pred_knn = knnClassifier.predict(X_test)
print("KNN prediction time: %s" % (time.time() - startTime))
evaluate(y_pred_knn)


# Logistic regression classifier
lrClassifier = LogisticRegression(random_state=1994, solver='lbfgs', multi_class='multinomial')
startTime = time.time()
lrClassifier.fit(X_train, y_train)
print("Logistic regression training time: %s" % (time.time() - startTime))
startTime = time.time()
y_pred_lr = lrClassifier.predict(X_test)
print("Logistic regression prediction time: %s" % (time.time() - startTime))
evaluate(y_pred_lr)


# SVM
svmClassifier = LinearSVC(random_state=1994) #gamma=0.01
startTime = time.time()
svmClassifier.fit(X_train, y_train)
print("SVM training time: %s" % (time.time() - startTime))
startTime = time.time()
y_pred_svm = svmClassifier.predict(X_test)
print("SVM prediction time: %s" % (time.time() - startTime))
evaluate(y_pred_svm)


# Evaluation
"""from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))"""


# Plotting the ROC curve
# To plot the ROC curve, we'll have to convert the classification problem to OneVsRest problem

y_test = [ i if (i==1) else 0 for i in y_test  ]

plt.figure()
plt.subplots(figsize=(8,6))

y_pred_dt = [ i if (i==1) else 0 for i in y_pred_dt  ]
fpr, tpr, _ = roc_curve(y_test, y_pred_dt, pos_label=1)
plt.plot(fpr, tpr, 'b', label="Decision tree, AUC=" + str(round(roc_auc_score(y_test, y_pred_dt), 3)))

y_pred_nb = [ i if (i==1) else 0 for i in y_pred_nb ]
fpr, tpr, _ = roc_curve(y_test, y_pred_nb)
plt.plot(fpr, tpr, 'r', label="Naive Bayes, AUC=" + str(round(roc_auc_score(y_test, y_pred_nb), 3)))

y_pred_rf = [ i if (i==1) else 0 for i in y_pred_rf  ]
fpr, tpr, _ = roc_curve(y_test, y_pred_rf)
plt.plot(fpr, tpr, 'g', label="Random Forest, AUC=" + str(round(roc_auc_score(y_test, y_pred_rf), 3)))

y_pred_lr = [ i if (i==1) else 0 for i in y_pred_lr  ]
fpr, tpr, _ = roc_curve(y_test, y_pred_lr)
plt.plot(fpr, tpr, 'm', label="Logistic Regression, AUC=" + str(round(roc_auc_score(y_test, y_pred_lr), 3)))

y_pred_svm = [ i if (i==1) else 0 for i in y_pred_svm  ]
fpr, tpr, _ = roc_curve(y_test, y_pred_svm)
plt.plot(fpr, tpr, 'y', label="SVM, AUC=" + str(round(roc_auc_score(y_test, y_pred_svm), 3)))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend()
plt.show()
