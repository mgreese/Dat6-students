# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:39:37 2015

@author: reesem
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from nltk import ConfusionMatrix

data = pd.read_csv('../data/zyx_prices.csv')
data.columns


column_list = ['ZYX1minPriceChange','ZYX5minPriceChange','ZYX10minPriceChange',
               'ZYX20minPriceChange','ZYX30minPriceChange','ZYX60minPriceChange','5fret',
               '10fret','20fret','30fret','60fret']

#Create new columns for each of the above columns. The new column indicates
#If it was positive or negatived

for thislist in column_list:
    data[thislist + 'binary'] = np.where(data[thislist] >= 0, 1, 0)
    

#Look at the data
data.plot(x='ZYX5minSentiment', y = '5fretbinary', kind = 'scatter')
data.plot(x='ZYX5minSentiment', y = '10fretbinary', kind = 'scatter')
data.plot(x='ZYX5minSentiment', y = '20fretbinary', kind = 'scatter')
data.plot(x='ZYX5minSentiment', y = '30fretbinary', kind = 'scatter')
data.plot(x='ZYX5minSentiment', y = '60fretbinary', kind = 'scatter')
#From what I see, 5min sentiment doesn't have much effect on fret.
#Let's try 60 min

data.plot(x='ZYX60minSentiment', y = '5fretbinary', kind = 'scatter')
data.plot(x='ZYX60minSentiment', y = '10fretbinary', kind = 'scatter')
data.plot(x='ZYX60minSentiment', y = '20fretbinary', kind = 'scatter')
data.plot(x='ZYX60minSentiment', y = '30fretbinary', kind = 'scatter')
data.plot(x='ZYX60minSentiment', y = '60fretbinary', kind = 'scatter')
#This doesn't look particularly useful either.

#Let's try just doing some regressions
Xsimple = data[['ZYX5minSentiment']]
ysimple = data['60fretbinary']
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(Xsimple, ysimple, test_size = .3)
#Convert back to dataframe
train = pd.DataFrame(data=X_train_simple, columns=['ZYX5minSentiment'])
train['60fretbinary'] = y_train_simple
test = pd.DataFrame(data=X_test_simple, columns=['ZYX5minSentiment'])
test['60fretbinary'] = y_test_simple

#Run a logistic regression
testfret_simple = LogisticRegression()
testfret_simple.fit(train[['ZYX5minSentiment']], y_train_simple)
B1 = testfret_simple.coef_[0][0]
B0 = testfret_simple.intercept_[0]
np.exp(B1)

testfret_simple.score(X_test_simple,y_test_simple)

preds_simple = testfret_simple.predict(X_test_simple)
print metrics.accuracy_score(y_test_simple, preds_simple)
print ConfusionMatrix(list(y_test_simple), list(preds_simple))


#Now look at more variables
X = data[[ 'ZYX10minSentiment', 'ZYX20minSentiment',
'ZYX30minSentiment', 'ZYX60minSentiment', 'ZYX10minTweets', 'ZYX20minTweets',
'ZYX30minTweets', 'ZYX60minTweets', 'ZYX10minPriceChange',
'ZYX20minPriceChange', 'ZYX30minPriceChange', 'ZYX60minPriceChange']]
y = data['60fretbinary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X = d[[ 'ZYX10minSentiment', 'ZYX20minSentiment',
'ZYX30minSentiment', 'ZYX60minSentiment', 'ZYX10minTweets', 'ZYX20minTweets',
'ZYX30minTweets', 'ZYX60minTweets', 'ZYX10minPriceChange',
'ZYX20minPriceChange', 'ZYX30minPriceChange', 'ZYX60minPriceChange']]
y = d['60fret']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)# Convert them back into dataframes, for convenience
train = pd.DataFrame(data=X_train, columns=[ 'ZYX10minSentiment', 'ZYX20minSentiment',
'ZYX30minSentiment', 'ZYX60minSentiment',  'ZYX10minTweets', 'ZYX20minTweets',
'ZYX30minTweets', 'ZYX60minTweets', 'ZYX10minPriceChange',
'ZYX20minPriceChange', 'ZYX30minPriceChange', 'ZYX60minPriceChange'])
train['60fret'] = y_train
test = pd.DataFrame(data=X_test, columns=[ 'ZYX10minSentiment', 'ZYX20minSentiment',
'ZYX30minSentiment', 'ZYX60minSentiment', 'ZYX10minTweets', 'ZYX20minTweets',
'ZYX30minTweets', 'ZYX60minTweets', 'ZYX10minPriceChange',
'ZYX20minPriceChange', 'ZYX30minPriceChange', 'ZYX60minPriceChange'])
test['60fret'] = y_test# create logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)# predict and calculate accuracy in one step
logreg.score(X_test, y_test) 
# Convert them back into dataframes, for convenience
train = pd.DataFrame(data=X_train, columns=[ 'ZYX10minSentiment', 'ZYX20minSentiment',
'ZYX30minSentiment', 'ZYX60minSentiment',  'ZYX10minTweets', 'ZYX20minTweets',
'ZYX30minTweets', 'ZYX60minTweets', 'ZYX10minPriceChange',
'ZYX20minPriceChange', 'ZYX30minPriceChange', 'ZYX60minPriceChange'])
train['60fretbinary'] = y_train
test = pd.DataFrame(data=X_test, columns=[ 'ZYX10minSentiment', 'ZYX20minSentiment',
'ZYX30minSentiment', 'ZYX60minSentiment', 'ZYX10minTweets', 'ZYX20minTweets',
'ZYX30minTweets', 'ZYX60minTweets', 'ZYX10minPriceChange',
'ZYX20minPriceChange', 'ZYX30minPriceChange', 'ZYX60minPriceChange'])
test['60fretbinary'] = y_test

# create logistic regression model
testfret = LogisticRegression()
testfret.fit(X_train, y_train)
# predict and calculate accuracy in one step
testfret.score(X_test, y_test)

preds = testfret.predict(X_test)
print ConfusionMatrix(list(y_test),list(preds))