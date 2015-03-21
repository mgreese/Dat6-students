# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 00:40:12 2015

@author: Brian
"""

import pandas as pd
import numpy as np
import matplotlib as plt

names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('imdb_top_10000.txt', delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()

dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

data['title'] = [t[0:-7] for t in data.title]
data.head()

data[['score', 'runtime', 'year', 'votes']].describe()

print len(data[data.runtime == 0])

data.runtime[data.runtime==0] = np.nan

data.runtime.describe()

plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")
remove_border()
# Received following message: 
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'module'object has not attribute 'hist'
# AND
# AttributeError: 'module'object has not attribute 'xlabel'

plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")
remove_border()
# Again, I'm receiving AttributeError messages. Is there an issue with the matplotlib that is not allowing me to produce a histogram?

plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")
remove_border()
# Again, I'm receiving AttributeError message.

data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})
# The genres were not listed alongside the counts

genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

decade =  (data.year // 10) * 10

tyd = data[['title', 'year']]
tyd['decade'] = decade

tyd.head()

decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
remove_border()
#Again, there were AttributeErrors when I tried to create graphs

grouped_scores = data.groupby(decade).score

mean = grouped_scores.mean()
std = grouped_scores.std()

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)
remove_border()
#Again, there were AttributeErrors when I tried to create graphs

for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values
    
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), 
                         tight_layout=True)

bins = np.arange(1950, 2013, 3)
for ax, genre in zip(axes.ravel(), genres):
    ax.hist(data[data[genre] == 1].year, 
            bins=bins, histtype='stepfilled', normed=True, color='r', alpha=.3, ec='none')
    ax.hist(data.year, bins=bins, histtype='stepfilled', ec='None', normed=True, zorder=0, color='#cccccc')
    
    ax.annotate(genre, xy=(1955, 3e-2), fontsize=14)
    ax.xaxis.set_ticks(np.arange(1950, 2013, 30))
    ax.set_yticks([])
    remove_border(ax, left=False)
    ax.set_xlabel('Year')
# More AttributeErrors and a NameError: 'axes' not defined
    
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), tight_layout=True)

bins = np.arange(30, 240, 10)

for ax, genre in zip(axes.ravel(), genres):
    ax.hist(data[data[genre] == 1].runtime, 
            bins=bins, histtype='stepfilled', color='r', ec='none', alpha=.3, normed=True)
               
    ax.hist(data.runtime, bins=bins, normed=True,
            histtype='stepfilled', ec='none', color='#cccccc',
            zorder=0)
    
    ax.set_xticks(np.arange(30, 240, 60))
    ax.set_yticks([])
    ax.set_xlabel("Runtime [min]")
    remove_border(ax, left=False)
    ax.annotate(genre, xy=(230, .02), ha='right', fontsize=12)
    
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), tight_layout=True)

bins = np.arange(0, 10, .5)

for ax, genre in zip(axes.ravel(), genres):
    ax.hist(data[data[genre] == 1].score, 
            bins=bins, histtype='stepfilled', color='r', ec='none', alpha=.3, normed=True)
               
    ax.hist(data.score, bins=bins, normed=True,
            histtype='stepfilled', ec='none', color='#cccccc',
            zorder=0)
    
    ax.set_yticks([])
    ax.set_xlabel("Score")
    remove_border(ax, left=False)
    ax.set_ylim(0, .4)
    ax.annotate(genre, xy=(0, .2), ha='left', fontsize=12)
    
# Fact One: With a score of 9.2, Shawshak Redemption was the #1 film of all time (and with good reason)
data.head(1)
    
# Fact Two: The average runtime was 103.6 minutes
data.runtime.mean()