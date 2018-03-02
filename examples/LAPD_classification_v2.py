# https://github.com/albertbup/deep-belief-network

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.9456&rep=rep1&type=pdf

# https://unbscholar.lib.unb.ca/islandora/object/unbscholar%3A8266
import getopt
import sys
import csv
import itertools
import os
import numpy as np
import nltk
np.random.seed(1337)  # for reproducibility
from sklearn.metrics.classification import accuracy_score
from keras.preprocessing import sequence
from dbn.tensorflow import SupervisedDBNClassification

print('Number of arguments: {} arguments.'.format(len(sys.argv)))
print('Argument List: {}'.format(str(sys.argv)))

try:
  iteration = 15;
  opts, args = getopt.getopt(sys.argv[1:], "i:", ["iteration="])
except getopt.GetoptError:
  print('LAPD_classification_v2.py -i <iteration_nb>')
  sys.exit(2)
for opt, arg in opts:
  if opt in ("-i", "--iteration"):
     iteration = arg


vocabulary_size = 1000
unknown_token = "UNKNOWN_TOKEN"
max_review_length = 250

# Open Save File
results = open('Results.csv', "w")
writer = csv.writer(results, delimiter='/', quoting=csv.QUOTE_NONE)
#header = ['TP', 'FN', 'FP', 'TN', 'FalsePROB', 'TruePROB']
header = [['F_prob', 'T_prob', 'Pred', 'Real']]
writer.writerows(header)

print("Fold {}\n".format(iteration))

# Create train and test files
for file in range(0,4):

    # Create the training data (Negative Examples)
    if (file == 0):
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',iteration ,'-train-lapd-no.neg')
    # Create the training data (Positive Examples)
    elif (file == 1):
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',iteration ,'-train-lapd-yes.pos')

    # Create the testing data (Negative Examples)
    elif (file == 2):
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',iteration ,'-test-lapd-no.neg')

    # Create the testing data (Positive Examples)
    else:
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',iteration ,'-test-lapd-yes.pos')

    # Read the data, append SENTENCE_START and SENTENCE_END tokens, and parse into sentences
    print("\nReading CSV file -> %s" % filename)
    with open(filename, 'r') as f:
        reader = csv.reader(f, skipinitialspace = True)

        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = ["%s" % (x) for x in sentences]

    print("Parsed %d sentences." % (len(list(sentences))))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # X_Train : Array of tokenized methods for training
    # y_Train : Array of labels for training
    # X_Test : Array of tokenized methods for testing
    # y_Test : Array of labels for testing

    # Create the training data (Negative Examples)
    if (file == 0):
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.transpose(np.asarray([0] * (len(list(sentences)))))

    # Create the training data (Positive Examples)
    elif (file == 1):
        X_train_yes = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train_yes = np.transpose(np.asarray([1] * (len(list(sentences)))))

        # Combine negative and positive examples
        X_Train = np.concatenate((X_train, X_train_yes), axis = 0)
        y_Train = np.concatenate((y_train, y_train_yes), axis = 0)

    # Create the testing data (Negative Examples)
    elif (file == 2):
        if (os.stat(filename).st_size != 0):
            X_Test = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            y_Test = np.transpose(np.asarray([0] * (len(list(sentences)))))

    # Create the testing data (Positive Examples)
    else:
        if (os.stat(filename).st_size != 0):
            X_Test = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            y_Test = np.transpose(np.asarray([1] * (len(list(sentences)))))

# Truncate and pad input sequences
X_Train = sequence.pad_sequences(X_Train, maxlen = max_review_length)
X_Test = sequence.pad_sequences(X_Test, maxlen = max_review_length)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[500,250,100],
                                             learning_rate_rbm=0.1,
                                             learning_rate=0.0001,
                                             n_epochs_rbm=50,
                                             n_iter_backprop=500,
                                             batch_size=16,
                                             activation_function='sigmoid',
                                             dropout_p=0.25)
classifier.fit(X_Train, y_Train)

# Test
Y_pred = classifier.predict(X_Test)
Y_p = classifier.predict_proba(X_Test)
Y_n = classifier.predict_proba_dict(X_Test)
print(Y_n)
print(Y_p)
print(Y_p)
print(Y_pred)
print(y_Test)
print('Done.\nAccuracy: %f' % accuracy_score(y_Test, Y_pred))
res = [[Y_p[0, 0], Y_p[0, 1], Y_pred, y_Test]]
writer.writerows(res)