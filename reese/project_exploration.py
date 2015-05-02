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
    data.rename(columns = {thislist : thislist.replace(' ','_')}, inplace = True) #Rename columns to eliminate spaces
data.describe()                                                                 #There is a massive variance. Let's try to standardize data

#Let's look at proportions of where adoptions are finalized in lieu of total numbers
data['total_intl'] = data['adoptions_finalized_abroad'] + data['adoptions_to_be_finalized_in_the_u.s.'] #Cumulative adoption 
data['intl_prop'] = data['adoptions_finalized_abroad'] / data['total_intl']     #Proportion of a country's adoptions finalized abroad
data['domestic_prop'] = data['adoptions_to_be_finalized_in_the_u.s.'] / data['total_intl'] #Proportion of a country's adoptions finalized domestically

#Explore the data
data.describe()

#Plot it as a histogram
plt.hist(data['total_intl'], bins=20)
plt.xlabel('Number of Adoptions')
plt.ylabel('Number of Countries')
plt.title('International Adoptions 2013')

#Not all countries follow the Hague Convention. Let's split them out.
data.isnull().sum()                                                             #3 columns have null values, representing non-Hague countries
hague_data = data[data['number_of_convention_cases'] > 0]                       #Generates a DataFrame for Hague Convention countries
hague_data.describe()                                                           #Looks like the proportion of Hague cases finalized abroad is a bit higher

#Hague Convention countries list median adoption 
hague_data[hague_data['median_asp_convention_adoption_fees'].isnull()]          #There are 3 null values for Greece, Guniea, and Togo
hague_data.median_asp_convention_adoption_fees.isnull().sum()                   
hague_data.median_asp_convention_adoption_fees.fillna(hague_data.median_asp_convention_adoption_fees.mean(), inplace=True) #Replace null values with the mean

#Let's look at fees. Why do some countries have generally higher fees?

#KMeans Cluster - let's look at a number of groupings.
#How many clusters should we use?

from sklearn import metrics
k_rng = range(2,10)
est_rng = [KMeans(n_clusters = k).fit(hague_data[['intl_prop','domestic_prop','average_days_to_completion',\
    'median_asp_convention_adoption_fees','total_intl']]) for k in k_rng]

silhouette_score = [metrics.silhouette_score(hague_data[['intl_prop','domestic_prop','average_days_to_completion',\
    'median_asp_convention_adoption_fees','total_intl']],q.labels_,metric = 'euclidean') for q in est_rng]
    
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng, silhouette_score, 'b*-')
plt.xlim([2,11])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.plot(5,silhouette_score[3], 'o', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

#Looks like the best number to use is 5

est = KMeans(n_clusters = 5,init='random')
est.fit(hague_data[['intl_prop','domestic_prop','average_days_to_completion','median_asp_convention_adoption_fees','total_intl']])
y_kmeans = est.predict(hague_data[['intl_prop','domestic_prop','average_days_to_completion','median_asp_convention_adoption_fees','total_intl']])
hague_data['predict'] = y_kmeans
hague_data[hague_data.predict == 0]
hague_data[hague_data.predict == 1]
hague_data[hague_data.predict == 2]
hague_data[hague_data.predict == 3]
hague_data[hague_data.predict == 4]


#Scatter plot
colors = np.array(['red','blue','yellow','green','pink'])
plt.figure(figsize=(5,5))

plt.subplot(221)
plt.scatter(hague_data['average_days_to_completion'],hague_data['median_asp_convention_adoption_fees'],c=colors[y_kmeans])
plt.xlabel('Avg Days to Completion')
plt.ylabel('Median Fees')

plt.subplot(222)
plt.scatter(hague_data['average_days_to_completion'],hague_data['intl_prop'],c=colors[y_kmeans])
plt.xlabel('Avg Days to Completion')
plt.ylabel('Proportion Finalized Abroad')

plt.subplot(223)
plt.scatter(hague_data['average_days_to_completion'],hague_data['domestic_prop'],c=colors[y_kmeans])
plt.xlabel('Avg Days to Completion')
plt.ylabel('Proportion Finalized Domestically')

plt.subplot(224)
plt.scatter(hague_data['average_days_to_completion'],hague_data['total_intl'],c=colors[y_kmeans])
plt.xlabel('Avg Days to Completion')
plt.ylabel('Total Number of Adoptions')
plt.ylim((0,210))

#It seems like fees are the main determinant in similarity, but that may be
#due to scaling. Let's see what happens when we scale & center fee data

#Scale the data
hague_data['scaled_fees'] = (hague_data['median_asp_convention_adoption_fees'] - \
    hague_data['median_asp_convention_adoption_fees'].mean()) / \
    hague_data['median_asp_convention_adoption_fees'].std()

#Test new silhouette coefficients
k_rng = range(2,10)
est_rng_scaled = [KMeans(n_clusters = k).fit(hague_data[['intl_prop','average_days_to_completion',\
    'scaled_fees','total_intl']]) for k in k_rng]

#Show the silhouette values
silhouette_score_scaled = [metrics.silhouette_score(hague_data[['intl_prop','average_days_to_completion',\
    'scaled_fees','total_intl']],q.labels_,metric = 'euclidean') for q in est_rng]

#Make a new elbow graph
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng, silhouette_score_scaled, 'b*-')
plt.xlim([2,11])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
#They're all negative now! This isn't good.

#Run a new analysis
est_scaled = KMeans(n_clusters = 5,init='random')
est_scaled.fit(hague_data[['intl_prop','average_days_to_completion','scaled_fees','total_intl']])
y_kmeans_scaled = est_scaled.predict(hague_data[['intl_prop','average_days_to_completion',\
    'scaled_fees','total_intl']])
    
#Graph it
plt.scatter(hague_data['average_days_to_completion'],hague_data['scaled_fees'], c = colors[y_kmeans_scaled])
plt.xlabel('Average Days to Completion')
plt.ylabel('Scaled Median Fees')
plt.title('Completion Time vs. Fees')

#Unable to determine any relationship based on the given variables.

"""
Domestic Adoption Analysis
"""

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
    
#Eliminate totals/
total_missing = ['Total','Missing']
for strings in states.columns:
    if any(x in strings for x in total_missing):
        del states[strings]

#Eliminate % signs & cast as floats
for this in states.columns:
    try:
        states[this] = states[this].str[:-1].astype(float)
    except:
        print strings + ' was unable to convert'
        
#Add back in a Total column
#Remember to eliminate the commas!
states['Total'] = age_2013['Total N'].str.replace(',','').astype(float)
        
#Drop the 'total' index
states.drop(states.index[52], inplace=  True)

#Make sure all columns are floats
for strings in states.columns:
    try:
        states[strings] = states[str(strings)].astype(float)
    except:
        try:
            states[strings] = states[strings].str.replace(',','').astype(float)
        except:
            print strings + ' was unable to convert'

#Look at some basic scatterplots

#Gender Distribution
plt.scatter(states.Male,states.Total)
plt.plot([states.Male.mean(),states.Male.mean()],[0,6000],linewidth=2, color = 'red')
plt.xlabel('Proportion of Male Adoptions')
plt.ylabel('Total Number of Adoptions')
plt.title('Male Adoptions in the US')

#Race Distribution
plt.figure(figsize = (8,8))
plt.title('Racial Distribution')

plt.subplot(221)
plt.scatter(states.White,states.Total)
plt.xlabel('% White Adoptions')
plt.ylabel('Total Adoptions')
plt.plot([states.White.mean(),states.White.mean()],[0,states.Total.max()],linewidth=2, color='red')

plt.subplot(222)
plt.scatter(states.Black,states.Total)
plt.xlabel('% Black Adoptions')
plt.ylabel('Total Adoptions')
plt.plot([states.Black.mean(),states.Black.mean()],[0,states.Total.max()],linewidth=2,color='red')

plt.subplot(223)
plt.scatter(states['Hispanic_(of_any_race)'],states.Total)
plt.xlabel('% Hispanic (of Any Race) Adoptions')
plt.ylabel('Total Adoptions')
plt.plot([states['Hispanic_(of_any_race)'].mean(),states['Hispanic_(of_any_race)'].mean()],[0,states.Total.max()],linewidth=2,\
    color='red')

plt.subplot(224)
plt.scatter(states.Asian,states.Total)
plt.xlabel('% Asian Adoptions')
plt.ylabel('Total Adoptions')
plt.plot([states.Asian.mean(),states.Asian.mean()],[0,states.Total.max()],linewidth=2,color='red')

'''

Issues to clean:
1) Eliminate empty spaces in column headers
2) Eliminate asterices in country names/define what they mean
3) Deal with NaN values in columns 3-5

'''
