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

for idx in range(len(description_str)):
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    description_str.iloc[idx]['Detailed Description'] = [word for word in description_str.iloc[idx]['Detailed Description'].split(' ') if word not in stopwords.words()];

#save data because it takes very long to remove stop words
pickle.dump(description_str, open('data_text.dat', 'wb'))
#get the words as an array for lda input
train_desc = [value[0] for value in description_str.iloc[0:].values];

#number of topics we will cluster for: 10
num_topics = 10;

# LDA
# We will use the gensim library for LDA. First, we obtain a id-2-word dictionary.
# For each headline, we will use the dictionary to obtain a mapping of the word id to their word counts. The LDA model uses both of these mappings.
id2word = gensim.corpora.Dictionary(train_desc);
corpus = [id2word.doc2bow(text) for text in train_desc];
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics);
lda.show_topic(0, topn=10)
# # generating LDA topics
# We will iterate over the number of topics, get the top words in each cluster and add them to a dataframe.
def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        # words = model.show_topic(i, topn = 15);
        words = model.show_topic(i, topn = 15);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);

get_lda_topics(lda, num_topics)

#visulization
# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
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
topics = lda.show_topics(num_topics=10,formatted=False)

fig, axes = plt.subplots(2, 5, figsize=(13,13), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=500)
    cloud.to_file('N.png')
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=17))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
