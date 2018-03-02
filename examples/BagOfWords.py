# /Users/noiseux1523/anaconda2/bin/python

# https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# ***************************************

import os

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LAPD-train.csv'), header=0, delimiter=";", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LAPD-test.csv'), header=0, delimiter=";", quoting=3)

    print 'The first LAPD is:'
    print train["LAPD"][0]

    # raw_input("Press Enter to continue...")


    print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    # nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_LAPD = []
    num_LAPD = train["LAPD"].size

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print "Cleaning and parsing the training set movie reviews...\n"
    # for i in xrange( 0, len(train["review"])):
    for i in xrange(0, len(train["LAPD"])):
        if ((i + 1) % 10 == 0):
            print "LAPD %d of %d\n" % (i + 1, num_LAPD)
        clean_train_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["LAPD"][i], False)))

    # ****** Create a bag of words from the training set
    #
    print "Creating the bag of words...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=1000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_LAPD)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    np.asarray(train_data_features)

    # ******* Train a random forest using the bag of words
    #
    print "Training the random forest (this may take a while)..."

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit(train_data_features, train["Problematic"])

    # Create an empty list and append the clean reviews one by one
    clean_test_LAPD = []

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0, len(test["LAPD"])):
        clean_test_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["LAPD"][i], False)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_LAPD)
    np.asarray(test_data_features)

    # Use the random forest to make sentiment label predictions
    print "Predicting test labels...\n"
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"LAPD": test["LAPD"], "Problematic": result})

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model.csv"
