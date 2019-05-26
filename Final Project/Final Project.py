# -*- coding: utf-8 -*-

""""
Created on Thu Mar 21 20:34:52 2019

@author: mtool
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import concurrent.futures
import time
import pyLDAvis.sklearn
from bs4 import BeautifulSoup 
import requests
import re
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
import nltk
import os
warnings.filterwarnings('ignore')



# Plotly based imports for visualization
import plotly
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
#plotly.tools.set_credentials_file(username=os.environ['PLOTLY_USERNAME'], api_key=os.environ['PLOTLY_API_KEY'])


# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
! python -m spacy download en_core_web_lg
#%%
#
# Define functions used later
#
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
 
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
#
# K-means function so we can iterate over number of clusters
#
def k_means(feature_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters
#%%
#
# Top 240 horror movies @ https://www.imdb.com/list/ls059633855/
#
debug = False
movieCode = set ([])
pages = ['1','2','3']
reviewAttrb = 'reviews?ref_=tt_ql_3'
movieAttrb = '?ref_=ttls_li_tt'
reviewDict = {}

#
# Get all URL's from page of 240 horror movies listing.
# Each movie has multiple pages, but we want the one that has the
# attribute ?ref_=ttls_li_tt.
# NOTE!!! The above algorithm broke. We now are searching for href attribute
# that conatin the text '/title/tt'. Each movie contains two of these, the
# second href attribute contains the movie title. We keep only that one
# 
# Once we find these, get the title of the
# movie and then update the URL to point to the review page by replacing
# the attribute with reviews?ref_=tt_ql_3
# Create a dictionary of the movie titles and the URLs to user review pages
# Note this URL points to the page with ALL the reviews for the movie
#
# I noticed the page with user reviews for each movie uses the same URL
# but just passes a different attribute after the movie number. I don't have
# to go to each movie page to find the review page, just subst the attribute
# at the end to get to the user review pages
#
for pgNum in pages:
    urlStr='https://www.imdb.com/list/ls059633855/?sort=list_order,asc&st_dt=&mode=detail&page='+pgNum
    page = requests.get(urlStr)
    soup = BeautifulSoup(page.text,'html.parser')
    print("Getting movie page URL's in page {}".format(pgNum))
    for a in soup.find_all('a', href=re.compile('/title/tt')):
        if a['href'] in set(movieCode):
            if str(a.contents[0]) =='\n': continue
            if debug:
                print ("Found the URL:", a['href'],'for the movie', a.contents[0])
            revURL = a['href']+(reviewAttrb)
            revURL = 'https://www.imdb.com'+ revURL
            title = str(a.contents[0])
            reviewDict [title] = revURL
        else:
            movieCode.add(a['href'])
print('\nTotal of URLs found = ', len(reviewDict.keys()))
#%%
#
# Find first permaLink and create a dictionary 
# {Title: permaLink} This page has movie rankings by user, etc...
#
permaDict ={}
print ("Getting PermaLinks...\n")
for k in reviewDict:
    reviewURL = reviewDict.get(k)
    title = str(k)                             # Save title for key in new Dict
    reviewPage = requests.get(reviewURL)
    reviewSoup = BeautifulSoup(reviewPage.text, 'html.parser')
    permaTag = reviewSoup.find('a', href=re.compile('/review/rw')) 
    if debug:
        print('PermaLink review URL:', permaTag['href'],'for movie',k)  
    permaURL = 'https://www.imdb.com'+ permaTag['href']
    permaDict [title] = permaURL    
#%%
#
# Retrieve the body of each each review and store in a dictionary
# {Title: Review Text}
#
textDict = {}
print('Getting review text... \n')
for k1 in permaDict:
    rawReviewText =''                           # Clear the buffer
    permaURL = permaDict.get(k1)
    title = str (k1)                            # Keep track of the movie title
    permaPage = requests.get(permaURL)
    permaSoup = BeautifulSoup(permaPage.text, 'html.parser')
    permaTag = permaSoup.find('div', {'class':'text show-more__control'})
    [p.replace_with(' ') for p in permaTag.findAll('br')]
    if debug:
        print(permaURL)
        print('Movie',k1, 'has text',permaTag.contents[0])
    for t in permaTag:
        rawReviewText = rawReviewText+ str(t)
    textDict[title] = rawReviewText


#%%
listOfReviews = []
listOfMovies = []
for k2 in textDict:
    listOfReviews.append(textDict.get(k2))
    listOfMovies.append(k2)

reviewDf = pd.DataFrame(
    {'Title': listOfMovies,
     'Review': listOfReviews
    })
reviewDf.head()
#%%
# Creating a spaCy object
nlp = spacy.load('en_core_web_lg')
#%%
doc = nlp(listOfReviews[0])
spacy.displacy.render(doc, style='ent',jupyter=True)
#%%
review = str(" ".join([i.lemma_ for i in doc]))
doc = nlp(review)
spacy.displacy.render(doc, style='ent',jupyter=True)
#%%
punctuations = string.punctuation
myStopWords = ['film','movie','films','movies']
stopwords = list(STOP_WORDS)+myStopWords
print("Number of Stop Words wrt to spaCy is: ", len(stopwords))
#%%
# POS tagging
for i in nlp(review):
    print(i,"=>",i.pos_)
#%%
 # Parser for reviews
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
#%%
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(reviewDf['processed_review'])
#%%
import seaborn as sns
words = vectorizer.get_feature_names()
sns.heatmap(pd.DataFrame(data_vectorized.todense(), columns=words), cmap='Blues')
plt.savefig('heatmap.png')
plt.gcf().set_size_inches(14, 8);
#%%
NUM_TOPICS = 5
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)
#%%
# Non-Negative Matrix Factorization Model
nmf = NMF(n_components=NUM_TOPICS)
data_nmf = nmf.fit_transform(data_vectorized) 
#%%
# Latent Semantic Indexing Model using Truncated SVD
lsi = TruncatedSVD(n_components=NUM_TOPICS)
data_lsi = lsi.fit_transform(data_vectorized)
#%%
# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 
#%%
# Keywords for topics clustered by Latent Dirichlet Allocation
print("LDA Model:")
selected_topics(lda, vectorizer)
#%%
# Keywords for topics clustered by Non-Negative Matrix Factorization
print("NMF Model:")
selected_topics(nmf, vectorizer)
#%%
# Keywords for topics clustered by Latent Semantic Indexing
print("LSI Model:")
selected_topics(lsi, vectorizer)
#%%
# Keywords for topics clustered by Latent Semantic Indexing
print("LSI Model:")
selected_topics(lsi, vectorizer)
#%%
pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(nmf, data_vectorized, vectorizer, mds='tsne')
dash
#%%
svd_2d = TruncatedSVD(n_components=2)
data_2d = svd_2d.fit_transform(data_vectorized)
#%%
import plotly
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.tools.set_credentials_file(username=['mtoolin'], api_key=['sisOh1qPTRxfq4SzLwVr'])
#%%
trace = go.Scattergl(
    x = data_2d[:,0],
    y = data_2d[:,1],
    mode = 'markers',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
    ),
    text = vectorizer.get_feature_names(),
    hovertext = vectorizer.get_feature_names(),
    hoverinfo = 'text' 
)
data = [trace]
iplot(data, filename='scatter-mode')
#%%
trace = go.Scattergl(
    x = data_2d[:,0],
    y = data_2d[:,1],
    mode = 'text',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
    ),
    text = vectorizer.get_feature_names()
)
data = [trace]
iplot(data, filename='text-scatter-mode')
#%%
def spacy_bigram_tokenizer(phrase):
    doc = parser(phrase) # create spacy object
    token_not_noun = []
    notnoun_noun_list = []
    noun = ""

    for item in doc:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            token_not_noun.append(item.text)
        if item.pos_ == "NOUN":
            noun = item.text
        
        for notnoun in token_not_noun:
            notnoun_noun_list.append(notnoun + " " + noun)

    return " ".join([i for i in notnoun_noun_list])
#%%
bivectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, ngram_range=(1,2))
bigram_vectorized = bivectorizer.fit_transform(reviewDf["processed_review"])
#%%
bi_lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_bi_lda = bi_lda.fit_transform(bigram_vectorized)
#%%
print("Bi-LDA Model:")
selected_topics(bi_lda, bivectorizer)
#%%
bi_dash = pyLDAvis.sklearn.prepare(bi_lda, bigram_vectorized, bivectorizer, mds='tsne')
bi_dash
