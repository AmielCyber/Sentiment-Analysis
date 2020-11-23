from sklearn.model_selection import train_test_split
import csv              # To read and split comma separated files or other delimiter
import numpy as np      # To use numpy arrays

def process_data():
    """ Process data for our data files.

    Process the data from our text files that contain reviews and the sentimental score for that review.
    Our text file contains the format: Review sentence \t sentimental score.

    :return: An input list of reviews and an output np array of the sentimental value (0 or 1).
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

    output_array = np.array(output_list)    # Make our output an np array to be compatible with sklearn functions

    ###########################################BOOK STUFF#####################################
    print('Number of documents in test data: {}'.format(len(input_list)))
    print('Samples per class (training): {}'.format(np.bincount(output_list)))
    print('end sentiment')
    ##########################################################################################

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


if __name__ == '__main__':
    # Process our data
    input_list, output_list = process_data()
    # Split our data
    X_train, X_test, y_train, y_test = split_data(input_list, output_list)

    print('hello')