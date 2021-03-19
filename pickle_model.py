# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:54:56 2021

@author: ADITYA RAJ YADAV
"""


import pandas as pd

df = pd.read_csv('balanced_reviews.csv')


df.dropna(inplace = True)
df = df[df['overall'] != 3]

import numpy as np
df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )


#features - reviewText col
#labels - Positivity

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 ) 


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(features_train)

features_train_vectorized = vect.transform(features_train)



#prepare the model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)


predictions = model.predict(vect.transform(features_test))


from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)



features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 ) 


from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df = 5).fit(features_train)


features_train_vectorized = vect.transform(features_train)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)


predictions = model.predict(vect.transform(features_test))


from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)

import pickle

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
    

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
pred = pickle_model.predict(vect.transform(features_test))

roc_auc_score(labels_test, pred)


#save count vectorize vocab
pickle.dump(vect.vocabulary_,open(
    'feature.pkl','wb'))
