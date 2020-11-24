from sklearn.model_selection import train_test_split            # To split our data
import csv                                                  # To read and split comma separated files or other delimiter
import numpy as np                                              # To use numpy arrays
from sklearn.feature_extraction.text import CountVectorizer     # To create bag of words
from sklearn.model_selection import cross_val_score             # To get a cross valid score
from sklearn.linear_model import LogisticRegression             # To train our data
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
                input_list.append(line[0])
                output_list.append(int(line[1]))

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


if __name__ == '__main__':
    # Process our data
    input_list, output_list = process_data()
    # Split our data
    X_train, X_test, y_train, y_test = split_data(input_list, output_list)
    # Set our input data into bag of words for feature extraction
    transformed_X_train = get_bag_of_words(X_train)
    scores = cross_val_score(LogisticRegression(), transformed_X_train, y_train, cv=5)
    print('Mean cross-validation accuracy {:.2f}'.format(np.mean(scores)))
    ################BOOK STUFF #######################################
    # In[12]
    vect = CountVectorizer().fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print('X_train:\n{}'.format(repr(X_train_transformed)))
    # In[13]
    feature_names = vect.get_feature_names()
    print('Number of features: {}'.format(len(feature_names)))
    print('First 20 features:\n{}'.format(feature_names[:20]))
    print('Features 1000 to 1030:\n{}'.format((feature_names[1000:1030])))
    print('Every 500th feature:\n{}'.format(feature_names[::500]))
    # In[14]
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train_transformed, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    print('Best parameters: ', grid.best_params_)
    # In[16]
    X_test_transformed = vect.transform(X_test)
    print('Test score: {:.2f}'.format(grid.score(X_test_transformed, y_test)))
    # In[17]
    vect = CountVectorizer(min_df=5).fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print('X_train with min_dif: {}'.format(repr(X_train_transformed)))
    # In[18]
    feature_names = vect.get_feature_names()
    print('First 50 features:\n{}'.format(feature_names[:50]))
    print('Features 900 to 1000:\n{}'.format(feature_names[900:1000]))
    print('Every 70th feature:\n{}'.format(feature_names[::70]))
    # In[19]
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train_transformed, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))
    # In 20
    print('Number of stop words: {}'.format(len(ENGLISH_STOP_WORDS)))
    print('Every 10th stopword:\n{}'.format(list(ENGLISH_STOP_WORDS)[::10]))
    # In[21]
    vect = CountVectorizer(min_df=5, stop_words='english').fit(X_train)
    X_train_transformed = vect.transform(X_train)
    print('X_train with stop words:\n{}'.format(repr(X_train_transformed)))
    # In[22]
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train_transformed, y_train)
    print('Best cross-validation score: {:.2f}'.format(grid.best_score_))




    #################################################################


    print('hello')