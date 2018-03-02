#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ****** Read the two training sets and the test set
#
import csv
import getopt

import pandas as pd
import os
import itertools
import sys
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
                                                         num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["LAPD"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


if __name__ == '__main__':

    try:
        iteration = 1;
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["iteration="])
    except getopt.GetoptError:
        print('LAPD_classification_v2.py -i <iteration_nb>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--iteration"):
            iteration = arg

    # Open Save File
    # results = open('Results.csv', "w")
    # writer = csv.writer(results, delimiter='/', quoting=csv.QUOTE_NONE)
    # header = ['TP', 'FN', 'FP', 'TN', 'FalsePROB', 'TruePROB']
    # header = [['F_prob', 'T_prob', 'Pred', 'Real']]
    # writer.writerows(header)

    print("Fold {}\n".format(iteration))

    # Initialize an empty list to hold the clean reviews
    clean_train_LAPD = []
    clean_test_LAPD = []
    clean_unlabeled_train_LAPD = []

    # Create train and test files
    for file in range(0, 4):

        # Create the training data (Negative Examples)
        if (file == 0):
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                       '-train-lapd-no.neg')

            # Read the data, append SENTENCE_START and SENTENCE_END tokens, and parse into sentences
            print("\nReading CSV file -> %s" % filename)
            with open(filename, 'r') as f:
                reader = csv.reader(f, skipinitialspace=True)

                # Split full comments into sentences
                sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
                sentences = ["%s" % (x) for x in sentences]
                sentences = [[x, 0] for x in sentences]

            print("Parsed %d sentences." % (len(list(sentences))))

            train = pd.DataFrame(sentences, columns=['LAPD', 'Problematic'])
            num_LAPD = train["LAPD"].size
            for i in xrange(0, len(train["LAPD"])):
                clean_train_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["LAPD"][i], False)))

        # Create the training data (Positive Examples)
        elif (file == 1):
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                       '-train-lapd-yes.pos')

            print("\nReading CSV file -> %s" % filename)
            with open(filename, 'r') as f:
                reader = csv.reader(f, skipinitialspace=True)

                # Split full comments into sentences
                sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
                sentences = ["%s" % (x) for x in sentences]
                sentences = [[x, 1] for x in sentences]

            print("Parsed %d sentences." % (len(list(sentences))))

            train_pos = pd.DataFrame(sentences, columns=['LAPD', 'Problematic'])
            num_LAPD = train_pos["LAPD"].size
            for i in xrange(0, len(train_pos["LAPD"])):
                clean_train_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train_pos["LAPD"][i], False)))

        # Create the testing data (Negative Examples)
        elif (file == 2):
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                       '-test-lapd-no.neg')

            print("\nReading CSV file -> %s" % filename)
            with open(filename, 'r') as f:
                reader = csv.reader(f, skipinitialspace=True)

                # Split full comments into sentences
                sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
                sentences = ["%s" % (x) for x in sentences]
                sentences = [[x, 0] for x in sentences]

            print("Parsed %d sentences." % (len(list(sentences))))

            test = pd.DataFrame(sentences, columns=['LAPD', 'Problematic'])
            num_LAPD = test["LAPD"].size
            for i in xrange(0, len(test["LAPD"])):
                clean_test_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["LAPD"][i], False)))

        # Create the testing data (Positive Examples)
        else:
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                       '-test-lapd-yes.pos')

            print("\nReading CSV file -> %s" % filename)
            with open(filename, 'r') as f:
                reader = csv.reader(f, skipinitialspace=True)

                # Split full comments into sentences
                sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
                sentences = ["%s" % (x) for x in sentences]
                sentences = [[x, 1] for x in sentences]

            print("Parsed %d sentences." % (len(list(sentences))))

            test_pos = pd.DataFrame(sentences, columns=['LAPD', 'Problematic'])
            num_LAPD = test_pos["LAPD"].size
            for i in xrange(0, len(test_pos["LAPD"])):
                clean_test_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test_pos["LAPD"][i], False)))


    filename = '/Users/noiseux1523/deep-belief-network/examples/data/OracleV5C14-training-dict.csv'

    # Read the data, append SENTENCE_START and SENTENCE_END tokens, and parse into sentences
    print("\nReading CSV file -> %s" % filename)
    with open(filename, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)

        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = ["%s" % (x) for x in sentences]

    print("Parsed %d sentences." % (len(list(sentences))))

    unlabeled_train = pd.DataFrame(sentences, columns=['LAPD'])
    num_LAPD = unlabeled_train["LAPD"].size
    for i in xrange(0, len(train["LAPD"])):
        clean_unlabeled_train_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(unlabeled_train["LAPD"][i], False)))

    train = train.append(train_pos, ignore_index=True)
    test = test.append(test_pos, ignore_index=True)


    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["LAPD"].size,
                                          test["LAPD"].size, unlabeled_train["LAPD"].size)

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in train["LAPD"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["LAPD"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # # Initialize and train the model (this will take some time)
    # print "Training Word2Vec model..."
    # model = Word2Vec(sentences, workers=num_workers, \
    #                  size=num_features, min_count=min_word_count, \
    #                  window=context, sample=downsampling, seed=1)
    #
    # # If you don't plan to train the model any further, calling
    # # init_sims will make the model much more memory-efficient.
    # model.init_sims(replace=True)
    #
    # # It can be helpful to create a meaningful model name and
    # # save the model for later use. You can load it later using Word2Vec.load()
    # model_name = "300features_1minwords_10context"
    # model.save(model_name)

    model = Word2Vec.load("300features_1minwords_10context")

    # model.doesnt_match("man woman child kitchen".split())
    # model.doesnt_match("france england germany berlin".split())
    # model.doesnt_match("paris berlin london austria".split())
    # model.most_similar("man")
    # model.most_similar("queen")
    # model.most_similar("awful")

    # ****** Create average vectors for the training and test sets
    #
    print "Creating average feature vecs for training reviews"

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print "Creating average feature vecs for test reviews"

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)

    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(trainDataVecs, train["Problematic"])

    # Test & extract results
    result = forest.predict(testDataVecs)

    # Write the test results
    output = pd.DataFrame(data={"LAPD": test["LAPD"], "Problematic": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print "Wrote Word2Vec_AverageVectors.csv"