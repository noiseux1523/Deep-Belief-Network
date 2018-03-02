# https://github.com/albertbup/deep-belief-network

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.9456&rep=rep1&type=pdf

# https://unbscholar.lib.unb.ca/islandora/object/unbscholar%3A8266

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import getopt
import sys
import csv
import itertools
import os
import numpy as np
import nltk
from nltk.corpus import stopwords

np.random.seed(1337)  # for reproducibility
from sklearn.metrics.classification import accuracy_score
from keras.preprocessing import sequence
from dbn.tensorflow import SupervisedDBNClassification

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

train = train.append(train_pos, ignore_index=True)
test = test.append(test_pos, ignore_index=True)

# ****** Create a bag of words from the training set
#
print "\nCreating the bag of words...\n"

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
test_data_features = vectorizer.transform(clean_test_LAPD)

# Numpy arrays are easy to work with, so convert the result to an
# array
np.asarray(train_data_features)
np.asarray(test_data_features)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[500, 250, 100],
                                         learning_rate_rbm=0.1,
                                         learning_rate=0.0001,
                                         n_epochs_rbm=50,
                                         n_iter_backprop=500,
                                         batch_size=16,
                                         activation_function='sigmoid',
                                         dropout_p=0)
classifier.fit(train_data_features.toarray(), train["Problematic"])

# Test
Y_pred = classifier.predict(test_data_features.toarray())
Y_p = classifier.predict_proba(test_data_features.toarray())
Y_n = classifier.predict_proba_dict(test_data_features.toarray())
print(Y_n)
print(Y_p)
print(Y_p)
print(Y_pred)
print(test["Problematic"])
print('Done.\nAccuracy: %f' % accuracy_score(test["Problematic"], Y_pred))
# res = [[Y_p[0, 0], Y_p[0, 1], Y_pred, test["Problematic"]]]
# writer.writerows(res)
