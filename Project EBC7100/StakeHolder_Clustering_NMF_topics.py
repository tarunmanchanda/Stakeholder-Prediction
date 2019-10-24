import nltk
import re
import numpy as np
import sys
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
import scipy as sp;
import sklearn;
import pickle;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pandas as pd
df = pd.read_csv('stakeholder.csv')
industry = df['Sector']
description_str = df[['Detailed Description']];
type(description_str)
# description_str = description.astype('str');
# type(description_str)


for idx in range(len(description_str)):
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    description_str.iloc[idx]['Detailed Description'] = [word for word in description_str.iloc[idx]['Detailed Description'].split(' ') if word not in stopwords.words()];
    #print logs to monitor output
    # if idx % 1000 == 0:
    #     sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(description_str)));

#save data because it takes very long to remove stop words
pickle.dump(description_str, open('data_text.dat', 'wb'))
#get the words as an array for lda input
train_desc = [value[0] for value in description_str.iloc[0:].values];

#number of topics we will cluster for: 15
num_topics = 15;

#NMF Starts here
#the count vectorizer needs string inputs, not array, so I join them with a space.
train_desc_sentences = [' '.join(text) for text in train_desc]
# Now, we obtain a Counts design matrix, for which we use SKLearn’s CountVectorizer module. The transformation will return a matrix of size (Documents x Features), where the value of a cell
# is going to be the number of times the feature (word) appears in that document.

vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(train_desc_sentences);
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
# And finally, obtain a NMF model, and fit it with the sentences.

#obtain a NMF model.
model = NMF(n_components=num_topics, init='nndsvd');
#fit the model
model.fit(xtfidf_norm)
feat_names = vectorizer.get_feature_names()
topicList = []
def get_nmf_topics(model, n_top_words):

    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()

    word_dict = {};
    for i in range(num_topics):

        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-15 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
        topicList.append(words)

    return pd.DataFrame(word_dict);

get_nmf_topics(model, 15)
finalTopic = []
for i in range(len(topicList)):
    temp = topicList[i]
    word = ''
    for words in temp:
        word = word + ' ' + words
    finalTopic.append(word.strip())


print(len(finalTopic))
#visulization
# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=15,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

# topics = model.
topics = finalTopic

fig, axes = plt.subplots(2, 5, figsize=(12,12), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = topics[i]
    cloud.generate(topic_words)
    # cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
