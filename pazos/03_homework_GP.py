# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 00:06:34 2015

@author: gpazos002
"""
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#tell pandas to display wide tables as pretty HTML tables
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
 #import data     
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('imdb_top_10000.txt', delimiter='\t', names=names).dropna()
data.describe()
print "Number of rows: %i" % data.shape[0]

#the worse movie every year
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values
    
# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

# highest rated movie that's passed year 2000 with votes above 200000
data[(data.votes > 200000) & (data.score >8) & (data.year >2000)] [['title', 'year', 'score', 'votes', 'genres']]

#plotted the density of total votes on a line graph
data.score.plot(kind='density', xlim=(0,10))
plt.savefig('vote_density.png')

#plotted the scatter of the number of votes per score
data.plot(kind='scatter', x='year', y='score', alpha=0.3)
plt.savefig('scatter_year_score.png')