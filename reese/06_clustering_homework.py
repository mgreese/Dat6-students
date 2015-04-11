'''
CLUSTER ANALYSIS ON COUNTRIES
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the data
d = pd.read_csv('../data/UNdata.csv')
np.random.seed(0)


# Run KMeans with k = 3
# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality
data = pd.DataFrame(d[['GDPperCapita','lifeMale','lifeFemale','infantMortality']])
est = KMeans(n_clusters = 3, init = 'random')
est.fit(data)
y_kmeans = est.predict(data)

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)
colors = np.array(['red','blue','green'])

plt.figure(figsize = (8,8))

plt.subplot(221)
plt.scatter(data['GDPperCapita'],data['lifeMale'], c = colors[y_kmeans])
plt.xlabel('GDP per Capita')
plt.ylabel('Male Life Expectancy')

plt.subplot(222)
plt.scatter(data['GDPperCapita'],data['lifeFemale'], c = colors[y_kmeans])
plt.xlabel('GDP per Capita')
plt.ylabel('Female Life Expectancy')

plt.subplot(223)
plt.scatter(data['GDPperCapita'],data['infantMortality'], c = colors[y_kmeans])
plt.xlabel('GDP per Capita')
plt.ylabel('Infant Mortality Rate')

# Print out the countries present within each cluster. Do you notice any general trend?
d['country_cluster'] = y_kmeans
d['country'][d['country_cluster'] == 0]
d['country'][d['country_cluster'] == 1]
d['country'][d['country_cluster'] == 2]
#Significantly more in cluster 1

# Print out the properties of each cluster. What are the most striking differences?
d[d['country_cluster'] == 0].describe()
d[d['country_cluster'] == 1].describe()
d[d['country_cluster'] == 2].describe()
d.groupby('country_cluster').describe()
# Advanced: Re-run the cluster analysis after centering and scaling all four variables 
#for each datapoint, subtract the mean (center) and divide by the standard deviation (scale)

# Advanced: How do the results change after they are centered and scaled? Why is this?
#GDP was on a different scale than the other measure. Standardizing it equalized influence.