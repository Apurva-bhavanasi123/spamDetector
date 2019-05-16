# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:09:49 2019
 
@author: Apoorva
"""

import sys
import nltk
import sklearn
import pandas as pd
import numpy as np
#load data 
sms=pd.read_table('SMSSpamCollection',header=None,encoding='utf-8')
sms.info()
sms.head(1)
classes=sms[0]
classes.value_counts()
#preprocessing data 
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
changed=encoder.fit_transform(sms[0])
changed[:10]
#store text messages i other array
text_messages=sms[1]
processed=text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','email')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','url')
processed = processed.str.replace(r'£|\$', 'dollar')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')
processed = processed.str.lower()
print(processed)
from nltk.corpus import stopwords
nltk.download("stopwords")
# remove stop words
stop_words=set(stopwords.words('english'))
processed=processed.apply(lambda x:' '.join(term for term in x.split() if term not in stop_words))
# remove words like -ing -ed
ps=nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
#generate bag of words
from nltk.tokenize import word_tokenize
bag_words=[]
for sent in processed:
    words=word_tokenize(sent)
    bag_words.extend(words)
bag_words
bag_words=nltk.FreqDist(bag_words)
feat=list(bag_words.keys())[:1500]
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in feat:
        features[word] = (word in words)

    return features

# Lets see an example!
features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print(key)
messages = zip(processed, changed)
seed = 1
np.random.seed = seed
#np.random.shuffle(list(messages))

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]
from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
train2=training
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
model = SklearnClassifier(SVC(kernel = 'linear'))
model.train(training)
model1=model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))

from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
#nltk_ensemble.train(train2)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))
txt_features, labels = zip(*testing)
prediction = model1.classify_many(txt_features)
print(classification_report(labels, prediction))
pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])