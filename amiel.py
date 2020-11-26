import mglearn                                                  # To use the visualizing tool
from networkx.drawing.tests.test_pylab import plt               # to visualize the cross-valid
from sklearn.model_selection import train_test_split            # To split our data
import csv                                                  # To read and split comma separated files or other delimiter
import numpy as np                                              # To use numpy arrays
from sklearn.feature_extraction.text import CountVectorizer     # To create bag of words
from sklearn.feature_extraction.text import TfidfVectorizer     # To use term frequency-inverse doc. freq. method
from sklearn.pipeline import make_pipeline                      # To use with tf-idf to ensure results are valid
from sklearn.model_selection import cross_val_score             # To get a cross valid score
from sklearn.linear_model import LogisticRegression             # To train our data
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
import nltk
import re                                                       # use regexp-based tokenization
from sklearn.model_selection import StratifiedShuffleSplit      # build a grid search using only 1% of the data as the training set
from sklearn.decomposition import LatentDirichletAllocation
def process_data():
    """ Process data for our data files.

    Process the data from our text files that contain reviews and the sentimental score for that review.
    Our text file contains the format: Review sentence \t sentimental score.

    :return: An input list of reviews and an output of an np array of the sentimental value (0 or 1).
    """
    # take list of filepaths to get data
    filepaths = {'amazon': 'amazon_cells_labelled.txt',
                 'yelp': 'yelp_labelled.txt',
                 'imdb': 'imdb_labelled.txt'}

    # populate input list and output list with data, separating sentences from the scores
    input_list = []  # Input list containing review sentences
    output_list = []  # Output list containing sentiment values of 0 or 1: 0 negative or 1 positive
    # Go through all the files and populate the input data along with its sentiment value in the output list
    for source, path in filepaths.items():
        # For all text files we have
        with open(path, 'r') as file:
            text = csv.reader(file, delimiter='\t')  # since our files are separated by a tab space
            for line in text:
                # For each line we will get a sentence and a sentiment value of 0(negative) and 1(positive)
                sentence = line[0]
                sentence = re.sub('\\t0', '', sentence)
                sentence = re.sub('\\t1', '', sentence)
                sentence = re.sub('\n', '', sentence)
                input_list.append(sentence)
                output_list.append(int(line[1]))

    #input_list = [doc.replace(r'(^[\t0\n]+|^[\t1\n]+(?=:))', " ") for doc in input_list]
    output_array = np.array(output_list)  # Make our output an np array to be compatible with sklearn functions

    return input_list, output_array


def split_data(X, y):
    """ Split data our data set of outputs and inputs.

    :param X:   The input list of our data set. In our case it will be sentences of a review.
    :param y:   The output list/array of our data set. In our case it will be the sentiment value based on a review.
    :return:    Four lists of:
                An input training set, an input test set, an output training set, and an output test set.
    """
    # Call sklearn.train_test_split to split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


def get_bag_of_words(input_text_list):
    """ get bag of words representation for data set of inputs (aka transforming the training data).
        These will help us train our data set easier.

    :param input_text_list: The input list that will like to have a bag-of-words representation.
    :return:
    """
    vect = CountVectorizer().fit(input_text_list)       # Create an instance of CountVectorizer
    transform_text_list = vect.transform(input_text_list)    # Transform the text list

    return transform_text_list


# define function to compare lemmatization in spacy with stemming in nltk
def compare_normalization(doc):
    # tokenize document in spacy
    doc_spacy = en_nlp(doc)
    # print lemmas found by spacy
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

# In[39]
# regexp used in CountVectorizer
regexp = re.compile('(?u)\\b\\w\\w+\\b')
# load spacy language model
en_nlp = spacy.load('en', disable=['parser', 'ner'])
old_tokenizer = en_nlp.tokenizer
# replace the tokenizer with the preceding regexp
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string))
# create a custom tokenizer using the spacy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]
# define a count vectorizer with the custom tokenizer
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

# In [43]





if __name__ == '__main__':

    # Process our data
    input_list, output_list = process_data()
    # Split our data
    X_train, X_test, y_train, y_test = split_data(input_list, output_list)
    print(input_list)
    # Set our input data into bag of words for feature extraction
    transformed_X_train = get_bag_of_words(X_train)
    scores = cross_val_score(LogisticRegression(), transformed_X_train, y_train, cv=5)
    print('Mean cross-validation accuracy {:.2f}'.format(np.mean(scores)))
    ################BOOK STUFF #######################################
    # In[12]
    print('In[12]')
    vect = CountVectorizer().fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print('X_train:\n{}'.format(repr(X_train_transformed)))
    # In[13]
    print('In[13]')
    feature_names = vect.get_feature_names()
    print('Number of features: {}'.format(len(feature_names)))
    print('First 20 features:\n{}'.format(feature_names[:20]))
    print('Features 1000 to 1030:\n{}'.format((feature_names[1000:1030])))
    print('Every 500th feature:\n{}'.format(feature_names[::500]))
    # In[14]
    print('In[14]')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train_transformed, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    print('Best parameters: ', grid.best_params_)
    # In[16]
    print('In[16]')
    X_test_transformed = vect.transform(X_test)
    print('Test score: {:.2f}'.format(grid.score(X_test_transformed, y_test)))
    # In[17]
    print('In[17]')
    vect = CountVectorizer(min_df=5).fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print('X_train with min_dif: {}'.format(repr(X_train_transformed)))
    # In[18]
    print('In[18]')
    feature_names = vect.get_feature_names()
    print('First 50 features:\n{}'.format(feature_names[:50]))
    print('Features 900 to 1000:\n{}'.format(feature_names[900:1000]))
    print('Every 70th feature:\n{}'.format(feature_names[::70]))
    # In[19]
    print('In[19]')
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train_transformed, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    """#############################Stopwords###########################################################"""
    # In[20]
    print('In[20]')
    print('Number of stop words: {}'.format(len(ENGLISH_STOP_WORDS)))
    print('Every 10th stopword:\n{}'.format(list(ENGLISH_STOP_WORDS)[::10]))
    # In[21]
    print('In[21]')
    vect = CountVectorizer(min_df=5, stop_words='english').fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print('X_train with stop words:\n{}'.format(repr(X_train_transformed)))
    # In[22]
    print('In[22]')
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train_transformed, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    """############################Rescaling the Data with tf-idf#########################################"""
    # In[23]
    print('In[23]')
    pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    # In[24]
    print('In[24]')
    vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
    # transform the training dataset
    X_train_transformed = vectorizer.transform(X_train)
    # find maximum value for each of the features over the dataset
    max_value = X_train_transformed.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()
    # get feature names
    feature_names = np.array(vectorizer.get_feature_names())
    print('Features with lowest tfidf:\n{}'.format(feature_names[sorted_by_tfidf[:20]]))
    print('Features with highest tfidf: \n{}'.format(feature_names[sorted_by_tfidf[-20:]]))
    # In[25]
    print('In[25]')
    sorted_by_idf = np.argsort(vectorizer.idf_)
    print('Features with lowest idf:\n{}'.format(feature_names[sorted_by_tfidf[:100]]))
    """##############################################Model Coefficients#################################"""
    # In[26]
    print('In[26]')
    mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps['logisticregression'].coef_, feature_names, n_top_features=40)
    """################################################Bag of Words with more than one word (n-grams)################"""
    # In[32]
    print('In[32]')
    pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
    # running the grid search takes a long time because of the
    # relatively large grid and the inclusion of trigrams
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
                  "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    print('Best parameters:\n{}'.format(grid.best_params_))
    # In[33]
    print('In[33]')
    # extract scores from grid_search
    scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
    # visualize heat map
    heatmap = mglearn.tools.heatmap(
        scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
        xticklabels=param_grid['logisticregression__C'],
        yticklabels=param_grid['tfidfvectorizer__ngram_range'])
    plt.colorbar(heatmap)
    # In[34]
    print('In[34]')
    # extract feature names and coefficients
    vect = grid.best_estimator_.named_steps['tfidfvectorizer']
    feature_names = np.array(vect.get_feature_names())
    coef = grid.best_estimator_.named_steps['logisticregression'].coef_
    mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
    # In[35]
    """
    print('35')
    # find 3-gram features
    mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
    # visualize only 3-gram features
    mglearn.tools.visualize_coefficients(coef.ravel()[mask], feature_names[mask], n_top_features=40)
    """
    """####################Advance Tokeniation,  Stemming, and Lemmatization#############################"""
    # In[36]
    print('In[36]')
    print('Spacy version: {}'.format(spacy.__version__))
    print('nltk version: {}'.format(nltk.__version__))
    # In[37]
    print('In[37]')
    # load spacy's English-language models
    en_nlp = spacy.load('en')
    # instantiate nltk's Porter stemmer
    stemmer = nltk.stem.PorterStemmer()
    # In[38]
    print('In[38]')
    compare_normalization(u'Our meeting today was worse than yesterday, ' 'I\'m scared of meeting the clients tomorrow.')
    # In[40]
    print('In[40]')
    # transform text_train using CountVectorizer with lemmatization
    X_train_lemma = lemma_vect.fit_transform(X_train)
    print("X_train_lemma.shape: {}".format(X_train_lemma.shape))

    # standard CountVectorizer for reference
    vect = CountVectorizer(min_df=5).fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print("X_train.shape: {}".format(X_train_transformed.shape))
    # In[41]
    print('In[41]')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99, train_size=0.01, random_state=0)
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
    # perform grid search with standard CountVectorizer
    grid.fit(X_train_transformed, y_train)
    print("Best cross-validation score " "(standard CountVectorizer): {:.3f}".format(grid.best_score_))
    # perform grid search with lemmatization
    grid.fit(X_train_lemma, y_train)
    print("Best cross-validation score " "(lemmatization): {:.3f}".format(grid.best_score_))
    # In[42]
    print('In[42]')
    vect = CountVectorizer(max_features=10000, max_df=.15)
    X = vect.fit_transform(X_train)
    # In[43]
    print('In[43]')
    lda = LatentDirichletAllocation(n_components=10, learning_method="batch", max_iter=25, random_state=0)
    # We build the model and transform the data in one step
    # Computing transform takes some time,
    # and we can save time by doing both at once
    document_topics = lda.fit_transform(X)
    # In[44]
    print('In[44]')
    print('lda.components_shape: {}'.format(lda.components_.shape))
    # In[45]
    print('In[45]')
    # For each topic (a row in the components_), sort the features (ascending)
    # Invert rows with [:, ::-1] to make sorting descending
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    # Get the feature names from the vectorizer
    feature_names = np.array(vect.get_feature_names())
    # In[46]
    # Print out the 10 topics:
    mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)
    # In 47
    print('In[47]')
    lda100 = LatentDirichletAllocation(n_components=100, learning_method="batch", max_iter=25, random_state=0)
    document_topics100 = lda100.fit_transform(X)
    # In 48
    print('In[48]')
    topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])

    sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    mglearn.tools.print_topics(topics=topics, feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=20)

    # In[49]
    print('In[49]')
    # sort by weight of "music" topic 45
    music = np.argsort(document_topics100[:, 45])[::-1]
    # print the five documents where the topic is most important
    #for i in music[:10]:
        # show first two sentences
    #    print(b".".join(X_train[i].split(b".")[:1]) + b".\n")

    # In[50]
    print('In[50]')
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    topic_names = ["{:>2} ".format(i) + " ".join(words)
                   for i, words in enumerate(feature_names[sorting[:, :2]])]
    # two column bar chart:
    for col in [0, 1]:
        start = col * 50
        end = (col + 1) * 50
        ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
        ax[col].set_yticks(np.arange(50))
        ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
        ax[col].invert_yaxis()
        ax[col].set_xlim(0, 2000)
        yax = ax[col].get_yaxis()
        yax.set_tick_params(pad=130)
    plt.tight_layout()

    print('end main')