# Sentiment-Analysis
A machine learning algorithm that will take user's review for a product  or a review for a movie and will predict if the review was negative or positive. 
import spacy
import numpy as np

text = open('amazon_cells_labelled.txt', 'r').read()
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
#Tokenizing sentences to break text down into sentences, words, or other units
#you’ll use word tokenization to separate the text into individual words. 
#First, you’ll load the text into spaCy, which does the work of tokenization for you
token_list = [token for token in doc]

#Stop words are words that may be important in human communication but are of little value for machines. 
#spaCy comes with a default list of stop words that you can customize. 
#For now, you’ll see how you can use token attributes to remove stop words:
filtered_tokens = [token for token in doc if not token.is_stop]

filtered_tokens
#Lemmatization seeks to address this issue.
#This process uses a data structure that relates all forms of a word back to its simplest form, 
#or lemma. Because lemmatization is generally more powerful than stemming, 
#it’s the only normalization strategy offered by spaCy.
lemmas = [
   f"Token: {token}, lemma: {token.lemma_}"
     for token in filtered_tokens
]

lemmas

#Vectorization is a process that transforms a token into a vector, or a numeric array that, 
#in the context of NLP, is unique to and represents various features of a token. 
#Vectors are used under the hood to find word similarities, classify text, and perform other NLP operations

filtered_tokens[1].vector


#--------------------------# 
#start to classfiy 
