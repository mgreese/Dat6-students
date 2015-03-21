# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


test =  pd.read_csv('./in/test.csv').dropna() 
train = pd.read_csv('./in/train.csv').dropna() 


train.sort_index(by='season',inplace=True,ascending=True) 
train.count =train.registered+train.casual

print "Number of rows: %i" % test.shape[0]


p1 = plt.bar(train.season, train.casual+train.registered,   width, color='r')
plt.xlabel("Season")
plt.ylabel("Total Demand")


