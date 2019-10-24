Python version used : 3.7
Conda version: 4.5.12

We have total of 8 files. 3 files are for Bag of Words to compare each algorithms with human marked labels, 3 are for TF-IDF to compare each algorithms with human marked labels, 1 file for Kappa with Bag of Words to make comparison between algorithms, and 1 file for Kappa with TF-IDF to make comparison between algorithms.

Files Included:
stakeholder.csv
stakholder_classification_SVM_TFIDF.py
stakholder_classification_Dtree_BOWtoTFIDF.py
Stakeholder_classification_lda.py
Stakeholder_Clustering_Kmeans.py
StakeHolder_Clustering_LDA_topics.py
StakeHolder_Clustering_NMF_topics.py



Below are the steps to run the code of clustering or classification for Bag of words to TF-IDF.

1. First download the dataset(stakeholder.csv) in the directory you are going to use.
2. Then download 3 files for classification, and 3 files for clustering.
3. Choose any file from the above mentioned names.
4. Then open the file in any of your IDE of python. (e.g. we used Spyder from Anaconda)
5. Run the code. You can use play button to run or select all text and use keyboard shortcut (shift+Enter)
6. for clustering files you may need to download pre request library if you get an error e.g. wordCloud library
7. for classification, code will run for the file you have chosen and will produce the output of different metrics along with visualization.
8. for clustering, code will run for the file you have chosen and will produce the output of different metrics along with visualization for predicting labels compared to human marked labels.


Note: after each run you need to remove the variables before running a new file. 
Also, these two files StakeHolder_Clustering_LDA_topics.py and StakeHolder_Clustering_NMF_topics.py might take longer time to run and form clusters.
