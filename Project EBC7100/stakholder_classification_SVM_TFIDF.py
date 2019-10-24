
"""
Created on Thu Apr  4 17:36:40 2019

@author: Preeti, Sofana, Tarun, raghar
"""

from sklearn.model_selection import cross_val_score
import numpy as np
import nltk 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

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
#np.unique(labels)

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
   
   


#Tf-IDF Model Implementation
# Creating the Tf-Idf model 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 300, min_df = 1, max_df = 0.3)
X = vectorizer.fit_transform(docs)
X = X.toarray()
#print(vectorizer.get_feature_names())

#transform to Tf-Idf
#Creating the Tf-Idf Model
#from sklearn.feature_extraction.text import TfidfTransformer
#transformer = TfidfTransformer()
#X = transformer.fit_transform(X).toarray()


   
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, labels, test_size=0.20, random_state=42, shuffle=True)

# fitting the model into machine learning algorithm
# Training first classifier
#SVC
from sklearn.svm import SVC
#cross validation to balance test and train
clf = SVC(kernel='linear', C=1.5, gamma='auto', random_state=42)
scores = cross_val_score(clf, x_train, y_train, cv=10)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))

# manual cross validation with shuffle
from sklearn.model_selection import StratifiedShuffleSplit
n_splits = 10
ssf = StratifiedShuffleSplit(n_splits, test_size=0.25, random_state=42)
# training the first classifer 
clf = SVC(kernel='linear', C=1.5, gamma='auto', random_state=42)
new_scores = []
X_array = X
labels = np.asarray(labels)
np.unique(labels)

from sklearn.metrics import accuracy_score
print("{} fold cross validation".format(n_splits))
for train_index, val_index in ssf.split(X_array, labels):
    x_train, y_train = X_array[train_index], labels[train_index]
    x_val, y_val = X_array[val_index], labels[val_index]
    clf.fit(x_train, y_train)
    #predicting the Test set results
    prediction_scores = clf.predict(x_val)
    print(accuracy_score(y_val, prediction_scores))
    #print ('yval-->',len(y_val))
    #np.unique(y_val)
    #np.unique(prediction_scores)
    #print ('prediction_scores-->',len(prediction_scores))
    new_scores.append(accuracy_score(y_val, prediction_scores))

print("Mean: {}".format(np.mean(new_scores)))

# Testing model performance (error analysis) 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, prediction_scores) 
    

#
#check = labels.tolist()  
#type(check) 
#for j in check:
#   count = check.count(j)
#   if count < 2:
#    print ('label index-->',check.index(j))
#    print ('label-->', j)
#    print(count)

#Accuracy recall and precision
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows
def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

print("label precision recall")
for label in range(15):
    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")

print("precision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))   

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

accuracy(cm)

# heatmap-visualization
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

Index= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15']
Cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15']

fig, ax = plt.subplots(figsize=(8,8))
df_cm = pd.DataFrame(cm, index = Index, columns = Cols)

sn.set(font_scale=1.4)
sn.heatmap(df_cm,annot=True, annot_kws={"size": 16},ax=ax)