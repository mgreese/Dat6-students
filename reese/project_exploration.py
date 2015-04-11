# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:43:10 2015

Adoption has had a significant impact on my life. When I was around 10 years old,
my mother adopted two 18 month old toddlers from Romania: Ileana and Traian. 

Traian came from an orphanage where the matron did not have time to change all 
the babies' diapers. Instead, she would paint them with disinfectant.

Ileana was marginally lucky; she came from a foster home. However, she was so
malnourished that she had developed rickets and spent the first few months in
her new home in a cast.

They are now both 16. Traian is studying to be a pilot. Ileana just broke up
with her first boyfriend, and she's pretty inconsolable, but she'll bounce back.
She wants to move to L.A. and be an actress.

I want to mine through international adoption statistics and find if there are
any telling correlations: high cost and high volume, number of adoptions per
thousand residents, ec.

@author: reesem
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from math import e
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets


"""
International Adoption Analysis

Divides countries into Hague & Non-Hague Participants

There is generally more info available for Hague Participants

"""
#Country adoption numbers
data = pd.read_csv('../../dat6-students/reese/adoption_countries.csv')
for thislist in data.columns:
    data.rename(columns = {thislist : thislist.replace(' ','_')}, inplace = True)
nan_data = data.fillna(value='not_applicable')
hague_data = nan_data[nan_data['number_of_convention_cases'] != 'not_applicable']

#China is an outlier, so let's ignore it for now.
hague_data_no_china = hague_data.set_index(hague_data['country'])
hague_data_no_china = hague_data_no_china.drop(hague_data_no_china['country']['China'])

#KMeans Cluster
est = KMeans(n_clusters = 3,init='random')
est.fit(hague_data[['adoptions_finalized_abroad','adoptions_to_be_finalized_in_the_u.s.','average_days_to_completion']])
y_kmeans = est.predict(hague_data[['adoptions_finalized_abroad','adoptions_to_be_finalized_in_the_u.s.','average_days_to_completion']])

nchina_est = KMeans(n_clusters = 3, init='random')
nchina_est.fit(hague_data_no_china[['adoptions_finalized_abroad','adoptions_to_be_finalized_in_the_u.s.','average_days_to_completion']])
nchina_y_kmeans = nchina_est.predict(hague_data_no_china[['adoptions_finalized_abroad','adoptions_to_be_finalized_in_the_u.s.','average_days_to_completion']])

#Scatter plot
colors = np.array(['red','blue','yellow'])
plt.figure()
plt.scatter(hague_data['average_days_to_completion'],hague_data['adoptions_finalized_abroad'],c=colors[y_kmeans])

#With no China
plt.figure()
plt.scatter(hague_data_no_china['average_days_to_completion'],hague_data_no_china['adoptions_finalized_abroad'],c=colors[nchina_y_kmeans])
plt.xlabel('Average Days to Completion')
plt.ylabel('Adoptions Finalized Abroad')

plt.figure()
plt.scatter(hague_data_no_china['average_days_to_completion'],hague_data_no_china['adoptions_to_be_finalized_in_the_u.s.'], c=colors[nchina_y_kmeans])
plt.xlabel('Average Days to Completion')
plt.ylabel('Adoptions to be Finalized in the U.S.')

#Maybe there's something to do with proportion adoption finalized domestic vs abroad?
hague_data_no_china['domestic_over_abroad'] = hague_data_no_china['adoptions_to_be_finalized_in_the_u.s.'] / (hague_data_no_china['adoptions_finalized_abroad'] + 1)
plt.figure()
plt.scatter(hague_data_no_china['average_days_to_completion'],hague_data_no_china['domestic_over_abroad'],c=colors[nchina_y_kmeans])
plt.xlabel('Average Days to Completion')
plt.ylabel('Proportion Domestic/Abroad')
#State census information
states = pd.read_csv('https://www.census.gov/popest/data/state/totals/2011/tables/NST-EST2011-01.csv')

#Number of international adoptions by state
statesadopt = pd.read_csv('../../dat6-students/reese/state_international_adoptions.csv')

#Percentage of domestic adoption cases that received some sort of subsidy
subsidy_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/adoption_subsidy2013.csv')

#Distribution of ages in domestic adoptions in 2013
age_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/final_age2013.csv')

#Domestic gender distribution 2013
gender_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/gender2013.csv')

#Data related to relatives involved with adoption/foster process in 2013
relative_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/prior_relations2013.csv')

#Breakdown of proportion of cases deemed 'special needs' in domestic adoption.
#Note 'special needs' is defined per state and not standardized across data
sneeds_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/special_needs2013.csv')

#Time between termination of parental rights and adoption finalization
tpr_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/tpr2013.csv')

#Domestic adoption race data 2013
race_2013 = pd.read_csv('../../dat6-students/reese/domestic_information/domestic_race2013.csv')

#Group all the dataframes into a list
newlist = [statesadopt, subsidy_2013, age_2013, gender_2013,relative_2013, sneeds_2013, tpr_2013, race_2013]

#Testing out merging
newframe = pd.merge(tpr_2013, sneeds_2013, on = 'State', how = 'left')

#Merge all the dataframes
states = pd.DataFrame(subsidy_2013.State)
for thislist in newlist:
    try:
        states = pd.merge(states, thislist, on = 'State', how = 'left')
    except:
        print thislist.head(1)
    #print thislist.head()

#Eliminate spaces in column headers    
for strings in states.columns:
    states.rename(columns = {strings : strings.replace(' ','_')}, inplace = True)

for df in newlist:
    for column in df:
        column.capitalize()

for strings in states.columns:
    try:
        print states[strings][30]
    except:
        print 'Didn\'t work'
#Get a sense of the data
data.head()
'''

Issues to clean:
1) Eliminate empty spaces in column headers
2) Eliminate asterices in country names/define what they mean
3) Deal with NaN values in columns 3-5

'''

#Replace empty spaces with _
for strings in data:
    data.rename(columns = {strings : strings.replace(' ','_')}, inplace = True)

data.average_days_to_completion.mean()
data.head(10)
data.number_of_convention_cases.isnull()

#Get a general graph of the data

#Is there a correlation between adoptions finalized abroad vs. in the US?
data.plot(x = ['Adoptions_Finalized_Abroad'], y = ['Adoptions_To_Be_Finalized_In_The_U.S.'], kind = 'scatter', xlim = (0,2500))
#Looks very L shaped. Slope looks undefined. Let's look closer
data.plot(x = ['adoptions_finalized_abroad'], y = ['adoptions_to_be_finalized_in_the_u.s.'], kind = 'scatter', xlim = (0,500))
#Still L shaped. Let's look closer again:
data.plot(x = ['adoptions_finalized_abroad'], y = ['adoptions_to_be_finalized_in_the_u.s.'], kind = 'scatter', xlim = (0,50),ylim=(0,50))
#Still L shaped. Looks like, generally, one precludes the other.

#Some of the country names have a * preceding them. According to the key with
#the original PDF, this means they do not participate in the Hague convention.
#Does this have any effect on where the adoption is completed?

#First make a numerical column. If a country has an asterisk, it gets 0.
#Otherwise, 1

def is_hague(row):
    if '*' not in row['country']:
        return 1
    else:
        return 0
        
data['is_hague'] = data.apply(lambda row: is_hague(row), axis = 1)

#We now have 2 dataframes: one for is_hague, one for isnt
data_is_hague = data[data['is_hague'] == 1]
data_not_hague = data[data['is_hague'] == 0]

#Let's plot the same graphs
plt.scatter(data_is_hague['adoptions_finalized_abroad'],data_is_hague['adoptions_to_be_finalized_in_the_u.s.'],color='red',alpha=.75)
plt.scatter(data_not_hague['adoptions_finalized_abroad'],data_not_hague['adoptions_to_be_finalized_in_the_u.s.'],color='blue',alpha=.75)
plt.xlim(0,500)
plt.ylim(0,350)
#There doesn't seem to be a substantial difference.

#What about size?
data_is_hague.shape
data_not_hague.shape

#Moving on to State data

#Clean up column headers
for strings in statesadopt:
    statesadopt.rename(columns = {strings : strings.replace(' ','_')}, inplace = True)
statesadopt.head()

#Preliminary plot
plt.scatter(statesadopt['adoptions_finalized_abroad'], statesadopt['adoptions_to_be_finalized_in_the_united_states'])
plt.xlabel('Adoptions Finalized Abroad')
plt.ylabel('Adoptions Finalized Domestically')
plt.xlim(-10, 200)
plt.ylim(-10,40)
#Need to standardize data. Maybe by adoption per 1k residents?