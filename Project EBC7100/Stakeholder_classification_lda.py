import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
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
check = labels.tolist()
type(check)
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
vectorizer = TfidfVectorizer(max_features=200, min_df=3, max_df=0.6) # You can still specify n-grams here.
X = vectorizer.fit_transform(docs)
X.toarray()

#using LDA algorithm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='svd',n_components=15)

# manual cross validation with shuffle
from sklearn.model_selection import StratifiedShuffleSplit
n_splits = 4
ssf = StratifiedShuffleSplit(n_splits, test_size=0.20, random_state=42)

new_scores = []
X_array = X.toarray()
labels = np.asarray(labels)

from sklearn.metrics import accuracy_score
print("{} fold cross validation".format(n_splits))
for train_index, val_index in ssf.split(X_array, labels):
   x_train, y_train = X_array[train_index], labels[train_index]
   x_val, y_val = X_array[val_index], labels[val_index]
   type(x_train)
   lda.fit(x_train, y_train)
   #predicting the Test set results
   prediction_scores = lda.predict(x_val)
   print(accuracy_score(y_val, prediction_scores))
   new_scores.append(accuracy_score(y_val, prediction_scores))

print("Mean: {}".format(np.mean(new_scores)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, prediction_scores)
print(cm)

#visualization
#from matplotlib.colors import ListedColormap
#import matplotlib.pyplot as plt
#X_set, y_set = x_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1 , stop = X_set[:, 0].max() + 200, step = 0.01),
#np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max() + 200, step = 0.01))
#plt.contourf(X1, X2, lda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#    alpha = 0.75, cmap = ListedColormap(sns.color_palette("Blues", 51)))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#    c = ListedColormap((‘red’, ‘green’, ‘blue’))(i), label = j)
#    plt.title(‘Logistic Regression (Training set)’)
#    plt.xlabel(‘LD1’)
#    plt.ylabel(‘LD2’)
#    plt.legend()
#    plt.show()
