#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:36:40 2019

@author: Preeti, Sofana, Tarun, raghar
"""

#importing libraries 
import nltk 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
from time import time
from sklearn import metrics
from scipy.stats import spearmanr


import pandas as pd
df = pd.read_csv('stakeholder.csv')
industry = df['Sector']
description = df['Detailed Description']
desplist = description.tolist()
categorical_labels = industry.tolist()
type(categorical_labels)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(categorical_labels)


#Data preparation and preprocess
def custom_preprocessor(text):
        text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations
        text =  re.sub(r'\s+',' ',text) #remove multiple spaces into a single space
        text = re.sub(r"\s+[a-zA-Z]\s+",' ',text) #remove a single character
        text = text.lower() 
        text = nltk.word_tokenize(text)       #tokenizing
        text = [word for word in text if not word in stop_words] #English Stopwords
        #text = [lemmatizer.lemmatize(word) for word in text]              #Lemmatising
        return text


docs = []


for item in desplist:
   t1 =  custom_preprocessor(item)
   var = ""
   for element in t1:
       var = var + ' ' + element
   docs.append(var.strip()) 
   


# Data transformation BOW
# Creating the BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.1)
X = vectorizer.fit_transform(docs)
X.toarray()

#transform to Tf-Idf
#Creating the Tf-Idf Model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


# k means determine k
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

distortions = []
K = range(1,14)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()



#**********************K-means*************************************************
name='BOW to TF-IDF'
t0 = time()

def bow_kmeans(X,k):
    from sklearn.cluster import KMeans
    estimator = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state= 0)
    estimator.fit(X)
    return estimator.labels_
    
#*****************************calculation**************************************
num_cluster = 10
clusters= bow_kmeans(X, k = num_cluster)

#print(len(clusters))
#print(len(labels))


print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tkappa\tcorr\tsilh_Clus\tsilh_HMN')
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%-9s\t%.3f\t%.3f'
          % (name, (time() - t0), 
             metrics.homogeneity_score(labels, clusters),
             metrics.completeness_score(labels, clusters),
             metrics.v_measure_score(labels, clusters),
             metrics.adjusted_rand_score(labels, clusters),
             metrics.adjusted_mutual_info_score(labels,  clusters),
             metrics.cohen_kappa_score(labels, clusters,weights='linear'),
             str(spearmanr(labels,clusters)),
             metrics.silhouette_score(X, clusters,
                                      metric='euclidean'),
             metrics.silhouette_score(X, labels,
                                      metric='euclidean'),
             ))

#**************************error analysis**************************************
from sklearn.metrics.cluster import contingency_matrix
x = labels #actual labels
y = clusters #predicted labels
error_analysis = contingency_matrix(x, y)


#**************************Plot************************************************
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
#
#from sklearn.datasets import make_moons
#X,y = make_moons(200, noise=.05, random_state=0)
#
#labels = KMeans(2, random_state=0).fit_predict(X)
#plt.scatter(X[:,0], X[:, 1], c=labels, s=50,cmap='viridis');

from sklearn.datasets import make_moons
X,Y = make_moons(200, noise=.05, random_state=0)
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=10, affinity='nearest_neighbors',
                           assign_labels='kmeans')
plottinglabels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=plottinglabels,
            s=50, cmap='viridis');
