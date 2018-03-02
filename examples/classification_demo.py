# https://github.com/albertbup/deep-belief-network

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.9456&rep=rep1&type=pdf

import tensorflow as tf
import csv
import itertools
import os
import numpy as np
import nltk
np.random.seed(1337)  # for reproducibility
from sklearn.metrics.classification import accuracy_score
from keras.preprocessing import sequence
from dbn.tensorflow import SupervisedDBNClassification

# Parameters
tf.flags.DEFINE_string("hidden_layers_structure", "2500,2500,2500", "Structure of the hidden layers")
tf.flags.DEFINE_float("learning_rate_rbm", 0.0001, "Learning rate of the restricted boltzmann machine (RBM)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of the deep belief network (ANN)")
tf.flags.DEFINE_integer("n_epochs_rbm", 150, "Number of epochs of the restricted boltzmann machine (RBM)")
tf.flags.DEFINE_integer("n_iter_backprop", 1000, "Number of epochs of the deep belief network (ANN)")
tf.flags.DEFINE_integer("batch_size", 32, "Training batch size")
tf.flags.DEFINE_string("activation_function", "sigmoid", "Activation function")
tf.flags.DEFINE_float("dropout_p", 0.5, "Dropout rate")

vocabulary_size = 2000
unknown_token = "UNKNOWN_TOKEN"
max_review_length = 500

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Open Save File
results = open('Results.csv', "w")
writer = csv.writer(results, delimiter='/', quoting=csv.QUOTE_NONE)
#header = ['TP', 'FN', 'FP', 'TN', 'FalsePROB', 'TruePROB']
header = [['F_prob', 'T_prob', 'Pred', 'Real']]
writer.writerows(header)

for it in range(0,50):
    print("Fold {}\n".format(it))

    # Create train and test files
    for file in range(0,4):

        # Create the training data (Negative Examples)
        if (file == 0):
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',it ,'-train-lapd-no.neg')
        # Create the training data (Positive Examples)
        elif (file == 1):
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',it ,'-train-lapd-yes.pos')

        # Create the testing data (Negative Examples)
        elif (file == 2):
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',it ,'-test-lapd-no.neg')

        # Create the testing data (Positive Examples)
        else:
            filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/',it ,'-test-lapd-yes.pos')

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
    classifier = SupervisedDBNClassification(hidden_layers_structure=list(map(int, FLAGS.hidden_layers_structure.split(","))),
                                             learning_rate_rbm=FLAGS.learning_rate_rbm,
                                             learning_rate=FLAGS.learning_rate,
                                             n_epochs_rbm=FLAGS.n_epochs_rbm,
                                             n_iter_backprop=FLAGS.n_iter_backprop,
                                             batch_size=FLAGS.batch_size,
                                             activation_function=FLAGS.activation_function,
                                             dropout_p=FLAGS.dropout_p)
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