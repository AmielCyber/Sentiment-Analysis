#!/usr/bin/env python
# coding: utf-8

# In[7]:


import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

english_stops = set(stopwords.words('english'))

#take list of filepaths to get data
filepaths = {'amazon': 'amazon_cells_labelled.txt',
               'yelp' : 'yelp_labelled.txt',
               'imdb': 'imdb_labelled.txt'}

#populate dataframes with data, separating sentences from the scores
dfs = []
for source, path in filepaths.items():
  df = pd.read_csv(path, names = ['review', 'sentiment'], sep = '\t')
  #df['source'] = source
  dfs.append(df)
df = pd.concat(dfs)
print(df)
print("-----------------------------------------------")
x_data = df['review']       # Reviews/Input
y_data = df['sentiment']    # Sentiment/Output

x_data = x_data.replace({'<.*?>': ''}, regex = True)          # remove html tag
x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)     # remove non alphabet
x_data = x_data.apply(lambda review: [w.lower() for w in review.split()])   # lower case
x_data = x_data.apply(lambda review: [w for w in review if w not in english_stops])  # remove stop words

print('Reviews')
print(x_data, '\n')
print('Sentiment')
print(y_data)

print("-----------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.33, shuffle = True)
print("X_train")
print("----")
print(X_train)
print("X_test")
print("----")
print(X_test)
print("y_train")
print("----")
print(y_train)
print("y_test")
print("----")
print(y_test)
print("x_train shape")
print((X_train).shape, '\n')
print("-----------------------------------------------")
#evaluating LogisticRegression using cross-validation
#cv = CountVectorizer()
#ctmTr = cv.fit_transform(X_train[:nwords].ravel()) 
#X_test_dtm = cv.transform(X_test.ravel())
#model = LogisticRegression()
#model.fit(ctmTr, y_train)
#y_pred_class = model.predict(X_test_dtm)
#accuracy_score(y_test, y_pred_class)
#scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
#print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))
def get_max_length():
    review_length = []
    for review in X_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))




# ENCODE REVIEW
token = Tokenizer(lower=False)    # no need lower, because already lowered the data in load_data()
token.fit_on_texts(X_train)
X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)

max_length = get_max_length()

# max_length = 200;
print("Length of review: ")
print(max_length)


X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test  = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1   # add 1 because of 0 padding
# vocab_size = len(t.word_index) + 1

print('Encoded X Train\n', X_train, '\n')
print('Encoded X Test\n', X_test, '\n')
print('Maximum review length: ', max_length)
print(X_train.shape)


# In[ ]:




