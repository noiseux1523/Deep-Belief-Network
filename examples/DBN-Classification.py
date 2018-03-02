import logging
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import getopt
import sys
import csv
import itertools
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
np.random.seed(1337)  # for reproducibility

########################################################################################################
# FUNCTIONS
########################################################################################################

# Function to generate dataframes based on input files
#    filename: Path to the input files
#    type: positive, negative or default (for unlabeled dataset)
def generateDatasets(filename, type):

    # Read the data, append SENTENCE_START and SENTENCE_END tokens, and parse into sentences
    print("\nReading CSV file -> %s" % filename)
    with open(filename, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)

        # Split full comments into sentences
        data = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        data = ["%s" % (x) for x in data]

        # Add label to dataset, if necessary
        if type.lower() == "positive":
            data = [[x, 1] for x in data]
            dataset = pd.DataFrame(data, columns=['LAPD', 'Problematic'])
        elif type.lower() == "negative":
            data = [[x, 0] for x in data]
            dataset = pd.DataFrame(data, columns=['LAPD', 'Problematic'])
        elif type.lower() == "default":
            dataset = pd.DataFrame(data, columns=['LAPD'])
        else:
            print("Wrong type of data or no type passed")
            sys.exit(2)

    print("Parsed %d data inputs." % (len(list(data))))
    return dataset

# Function to create a single feature vector
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

# Function to create average word vectors
#
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

# Function to clean the input data
#
def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["LAPD"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

########################################################################################################
# PARAMETERS
########################################################################################################

# Pass parameter values
try:
    # Default Initialization
    iteration=1
    word_embedding=True
    w2v_train_file='None' # "/Users/noiseux1523/deep-belief-network/examples/data/OracleV5C14-training-dict.csv"
    w2v_dictionary='None' # "/Users/noiseux1523/deep-belief-network/examples/Trained_dictionary"
    embedding_dimension=300
    min_word_count=1
    context=10
    downsampling=0.001
    hidden_layers_units=[500,250,100]
    learning_rate_rbm=0.1
    learning_rate_backprop=0.0001
    n_epochs_rbm=50
    n_iter_backprop=500
    batch_size=16
    activation_function="sigmoid"
    dropout_p=0

    # Reading arguments
    opts, args = getopt.getopt(sys.argv[1:], "", ['iteration=',
                                                  'word-embedding=',
                                                  'w2v-train-file=',
                                                  'w2v-dictionary=',
                                                  'embedding-dimension=',
                                                  'min-word-count=',
                                                  'context=',
                                                  'downsampling=',
                                                  'hidden-layers-units=',
                                                  'learning-rate-rbm=',
                                                  'learning-rate-backprop=',
                                                  'n-epochs-rbm=',
                                                  'n-iter-backprop=',
                                                  'batch-size=',
                                                  'activation-function=',
                                                  'dropout-p=',])

except getopt.GetoptError:
    print('RNN-complete.py \n'
           '\t--iteration <int> \n'
           '\t--word-embedding <string> \n'
           '\t--w2v-train-file <string> \n'
           '\t--w2v-dictionary <string> \n'
           '\t--embedding-dimension <int> \n'
           '\t--min-word-count <int> \n'
           '\t--context <int> \n'
           '\t--downsampling <float> \n'
           '\t--hidden-layers-units <array> \n'
           '\t--learning-rate-rbm <float> \n'
           '\t--learning-rate-backprop <float> \n'
           '\t--n-epochs-rbm <int> \n'
           '\t--n-iter-backprop <int> \n'
           '\t--batch-size <int> \n'
           '\t--activation-function <string> \n'
           '\t--dropout-p <float> \n')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--iteration"):
        iteration = arg
    elif opt in ("--word-embedding"):
        word_embedding = arg
    elif opt in ("--w2v-train-file "):
        w2v_train_file = arg
    elif opt in ("--w2v-dictionary"):
        w2v_dictionary = arg
    elif opt in ("--embedding-dimension"):
        embedding_dimension = int(arg)
    elif opt in ("--min-word-count"):
        min_word_count = float(arg)
    elif opt in ("--context"):
        context = int(arg)
    elif opt in ("--downsampling"):
        downsampling = float(arg)
    elif opt in ("--hidden-layers-units"):
        hidden_layers_units = list(map(int, arg.split(",")))
    elif opt in ("--learning-rate-rbm"):
        learning_rate_rbm = float(arg)
    elif opt in ("--learning-rate-backprop"):
        learning_rate_backprop = float(arg)
    elif opt in ("--n-epochs-rbm"):
        n_epochs_rbm = int(arg)
    elif opt in ("--n-iter-backprop"):
        n_iter_backprop = int(arg)
    elif opt in ("--batch-size"):
        batch_size = int(arg)
    elif opt in ("--activation-function"):
        activation_function = arg
    elif opt in ("--dropout-p"):
        dropout_p = float(arg)
    else:
        print("{} is an invalid parameter".format(opt))

########################################################################################################
# GENERATE DATASETS
########################################################################################################

# Initialize an empty list to hold the clean reviews
clean_train_LAPD = []
clean_test_LAPD = []
clean_unlabeled_LAPD = []

# Create train and test files
print("Processing Fold {} ...\n".format(iteration))

for file in range(0, 4):

    # Create the training data (Negative Examples)
    if (file == 0):
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                   '-train-lapd-no.neg')
        train = generateDatasets(filename, "negative")
        for i in xrange(0, len(train["LAPD"])):
            clean_train_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["LAPD"][i], False)))

    # Create the training data (Positive Examples)
    elif (file == 1):
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                   '-train-lapd-yes.pos')
        train_pos = generateDatasets(filename, "positive")
        for i in xrange(0, len(train_pos["LAPD"])):
            clean_train_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train_pos["LAPD"][i], False)))

    # Create the testing data (Negative Examples)
    elif (file == 2):
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                   '-test-lapd-no.neg')
        test = generateDatasets(filename, "negative")
        for i in xrange(0, len(test["LAPD"])):
            clean_test_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["LAPD"][i], False)))

    # Create the testing data (Positive Examples)
    else:
        filename = "{}{}{}".format('/Users/noiseux1523/deep-belief-network/OracleV5C14-training/', iteration,
                                   '-test-lapd-yes.pos')
        test_pos = generateDatasets(filename, "positive")
        for i in xrange(0, len(test_pos["LAPD"])):
            clean_test_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test_pos["LAPD"][i], False)))

########################################################################################################
# GENERATE MODEL
########################################################################################################

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=hidden_layers_units,
                                         learning_rate_rbm=learning_rate_rbm,
                                         learning_rate=learning_rate_backprop,
                                         n_epochs_rbm=n_epochs_rbm,
                                         n_iter_backprop=n_iter_backprop,
                                         batch_size=batch_size,
                                         activation_function=activation_function,
                                         dropout_p=dropout_p)

########################################################################################################
# GENERATE WORD EMBEDDING
########################################################################################################

# Use word embeddings
if word_embedding == 'True':
    if w2v_dictionary != "None":
        # Load w2v model
        print "Loading w2v model..."
        model = Word2Vec.load(w2v_dictionary)

    elif w2v_train_file != "None":
        # Create the word embeding data (Unlabeled Examples)
        unlabeled_train = generateDatasets(w2v_train_file, "default")
        for i in xrange(0, len(train["LAPD"])):
            clean_unlabeled_LAPD.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(unlabeled_train["LAPD"][i], False)))

        # Verify the number of examples that were read
        train = train.append(train_pos, ignore_index=True)
        test = test.append(test_pos, ignore_index=True)
        print "Read %d labeled train reviews, %d labeled test reviews, " \
              "and %d unlabeled reviews\n" % (train["LAPD"].size,
                                              test["LAPD"].size,
                                              unlabeled_train["LAPD"].size)

        # Load the punkt tokenizer
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Split the labeled and unlabeled training sets into clean sentences
        sentences = []  # Initialize an empty list of sentences

        print "Parsing sentences from training set ..."
        for review in train["LAPD"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

        print "Parsing sentences from unlabeled set ..."
        for review in unlabeled_train["LAPD"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

        # Set parameters and train the word2vec model
        #
        # Import the built-in logging module and configure it so that Word2Vec
        # creates nice output messages
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

        # Initialize and train the model (this will take some time)
        print "Training Word2Vec model..."
        model = Word2Vec(sentences, workers=4,
                         size=embedding_dimension, min_count=min_word_count,
                         window=context, sample=downsampling, seed=1)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        print "Model has finished training, it was saved under the name: Trained_dictionary"
        model_name = "Trained_dictionary"
        model.save(model_name)

    else:
        print("Missing argument to --w2v-dictionary if you want use a pretrain w2v model "
              "or to --w2v-train-file if you want to train a w2v model with a specified training file")
        sys.exit(2)

    # Create average vectors for the training and test sets
    print "Creating average feature vecs for training reviews ..."
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, embedding_dimension)

    print "Creating average feature vecs for test reviews ..."
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, embedding_dimension)

    ###################
    # TRAIN THE MODEL #
    ###################
    classifier.fit(trainDataVecs, train["Problematic"])

# Use bag-of-word
else:
    # Append negative and positive examples
    train = train.append(train_pos, ignore_index=True)
    test = test.append(test_pos, ignore_index=True)

    # Create a bag of words from the training set
    print "\nCreating the bag of words...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=1000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_LAPD)
    test_data_features = vectorizer.transform(clean_test_LAPD)

    # Numpy arrays are easy to work with, so convert the result to an array
    np.asarray(train_data_features)
    np.asarray(test_data_features)

    ###################
    # TRAIN THE MODEL #
    ###################
    classifier.fit(train_data_features.toarray(), train["Problematic"])

########################################################################################################
# EVALUATE THE MODEL
########################################################################################################

Y_pred = classifier.predict(testDataVecs)
Y_p = classifier.predict_proba(testDataVecs)
Y_n = classifier.predict_proba_dict(testDataVecs)
print(Y_n)
print(Y_p)
print(Y_p)
print(Y_pred)
print(test["Problematic"])
# print('Done.\nAccuracy: %f' % accuracy_score(test["Problematic"], Y_pred))
# res = [[Y_p[0, 0], Y_p[0, 1], Y_pred, test["Problematic"]]]
# writer.writerows(res)
