# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:35:59 2019

@author: mtool
"""
#%%
#import pattern.en
import re
import string
from nltk.cluster.util import cosine_distance
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#%%
# Supporting functions, taken from Sarkar
#
def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

def bow_extractor(corpus, ngramRange=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngramRange)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def build_feature_matrix(documents, feature_type='frequency'):
    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, 
                                     ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)    
    return vectorizer, feature_matrix

def compute_cosine_similarity(doc_features, corpus_features,
                              top_n=1):
    # get document vectors
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
    # compute similarities
    similarity = np.dot(doc_features, 
                        corpus_features.T)
    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 2))
                            for index in top_docs]
    return top_docs_with_score
#%%
import nltk
nltk.download('wordnet') # first-time use only
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
#%%

AmazonBookList =  ['All the Ugly and Wonderful Things: A Novel',
                   'The Tuscan Child',
                   'Where the Crawdads Sing',
                   'The Nightingale: A Novel',
                   'The Goldfinch: A Novel (Pulitzer Prize for Fiction)',
                   'The Life We Bury',
                   'All the Light We Cannot See: A Novel',
                   'Spilled Milk: Based on a true story',
                   'What Alice Forgot',
                   'The Flight Attendant: A Novel',
                   'Winter Garden',
                   "The Storyteller's Secret: A Novel",
                   'Ordinary Grace: A Novel',
                   'All the Ugly and Wonderful Things: A Novel',
                   'It Ends with Us: A Novel',
                   'The Shack: Where Tragedy Confronts Eternity',
                   'Beneath a Scarlet Sky: A Novel',
                   'Before We Were Yours: A Novel',
                   'Small Great Things: A Novel',
                   "The Boy on the Wooden Box: How the Impossible Became Possible . . . on Schindler's List",
                   'A Man Called Ove: A Novel',
                   "The Ladies' Room",
                   'The Butterfly Garden (The Collector Book 1)',
                   'HOSTILE WITNESS: A Josie Bates Thriller (The Witness Series Book 1)']


#
stopword_list = ['a','the','and','novel','s']

GoogleResults = [{'All the Ugly and Wonderful Things: A Novel':["All the Ugly\
                and Wonderful Things: A Novel Kindle Edition by Bryn Greenwood \
                A powerful novel you won't soon forget, Bryn Greenwood's All\
                the Ugly and Wonderful Things challenges all we know and\
                believe about love. A beautiful and provocative love story\
                between two unlikely people and the hard-won relationship that\
                elevates them above the Midwestern meth lab backdrop of their\
                lives.",
                "Review: Love is complicated in Ugly and Wonderful Things ..']},
     {}]
#%%


#
# Create tokenized list of each title
#
tokenizer = RegexpTokenizer(r'\w+')
for title in AmazonBookList:
    tokens = []
    title = title.lower()
    tokens = word_tokenize(title)
    tokens = remove_characters_after_tokenization(tokens)
    filteredTokens = [token for token in tokens if token not in stopword_list]      
#    tfidf_vectorizer, tfidf_features = build_feature_matrix(filteredTokens,
#                                                        feature_type='tfidf')
    AmazonTokens.append(filteredTokens)  
#%%
   
#%%
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopword_list)
def cos_similarity(textlist):
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()
titleMatrix = cos_similarity(AmazonBookList) 
titleArray = np.asarray(titleMatrix).reshape(-1)
titleArray.sort().unique()


    

    