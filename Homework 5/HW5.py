# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:30:48 2019

@author: mtool
"""
import sys; print ("Python", sys.version)
from nltk.corpus import stopwords
import nltk
from nltk import Tree
from nltk import word_tokenize, pos_tag,ne_chunk
from nltk import RegexpParser

from pattern.en import tag, parsetree, Chunk
from bs4 import BeautifulSoup 
import requests
from difflib import SequenceMatcher
import re
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#%%
#
# Top 240 horror movies @ https://www.imdb.com/list/ls059633855/
#
debug = 1
pages = ['1', '2', '3']
reviewAttrb = 'reviews?ref_=tt_ql_3'
movieAttrb = '?ref_=ttls_li_tt'
reviewDict = {}

#
# Get all URL's from page of 240 horror movies listing.
# Each movie has multiple pages, but we want the one that has the
# attribute ?ref_=ttls_li_tt.  Once we find these, get the title of the
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
    for a in soup.find_all('a', href=re.compile('/?ref_=ttls_li_tt')):
        if debug:
            print ("Found the URL:", a['href'],'for the movie', a.contents[0])
        revURL = a['href'].replace(movieAttrb, reviewAttrb)
# In case I need a list of dictionaries
#        listOfURL.append({a.contents[0]:'https://www.imdb.com'+revURL})
        revURL = 'https://www.imdb.com'+ revURL
        title = str(a.contents[0])
        reviewDict [title] = revURL
print('Total of URLs found = ', len(reviewDict.keys()))
#%%
#
# Find first permaLink and create a dictionary 
# {Movie title:permaLink} This page has movie rankings by user, etc...
#
permaDict ={}
for k in reviewDict:
    reviewURL = reviewDict.get(k)
    title = str(k)                             # Save title for key in new Dict
    reviewPage = requests.get(reviewURL)
    reviewSoup = BeautifulSoup(reviewPage.text, 'html.parser')
#    for a in reviewSoup.find_all('a', href=re.compile('review/rw')):
#        if debug:
#           print('Found reveiw URL:', a['href'],'for movie', k)
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
#
# Now that all the reviews are gathered, chunk each one and create a
# dictionary with {Title:[(chunk1, chunk2,...,chunkn)]}
#
nounPhraseDict = {}
for k2 in textDict:
    rawText = textDict.get(k2)
    str_tokens = nltk.word_tokenize(rawText)   # not sure if we need this.
    tree = parsetree(rawText)
    print (k2)
    for sentence_tree in tree:
        for chunk in sentence_tree.chunks:
            if (chunk.type =='NP'):
                nounPhrase = [(word.string, word.type) for word in chunk.words]
                print(nounPhrase)
                nounPhraseDict.setdefault(k2, []).append(nounPhrase)
                
#%%
# 
str = "The quick brown fox jumped over the lazy dog"               
#str = textDict.get('The Wailing')
str_tokens = nltk.word_tokenize(str)
tree = parsetree(str)
for sentence_tree in tree:
    for chunk in sentence_tree.chunks:
        print(chunk.type, '->', [(word.string, word.type) for word in chunk.words])

#%%
#
# Some functions from Sarkar book maybe for later use
#

# Create a shallow parsed sentence tree
def createSentenceTree (sentence, lemmatize = False):
    sentence_tree = parsetree(sentence, relations=True, lemmata=lemmatize)
    return sentence_tree

# Get various constituents of the parse tree
def getSentenceTreeConstituents (sentence_tree):
    return sentence_tree.constituents()

# Process the shallow parsed tree inot an easy to understand format
def processSentenceTree(sentence_tree):
    tree_constituents = getSentenceTreeConstituents(sentence_tree)
    processedTree = [(item.type,
                     [(w.string, w.type) for w in item.words])
                      if type(item) == Chunk
                      else ('-',[(item.string, item.type)])
                      for item in tree_constituents]
    return processedTree

# Print the sentence tree using nltk's Tree syntax
def printSentenceTree (sentence_tree):
    processed_tree = processSentenceTree(sentence_tree)
    processed_tree = [Tree(item[0],[Tree(x[1], [x[0]]) for x in item[1]]) for item in processed_tree]
    tree = Tree('S', processed_tree)
    print (tree)
    
# Visualize the sentence tree using nltk's Tree syntax
def visualizeSentenceTree(sentence_tree):
    processed_tree = processSentenceTree(sentence_tree)
    processed_tree = [Tree(item[0],[Tree(x[1], [x[0]]) for x in item[1]]) for item in processed_tree]
    tree = Tree('S', processed_tree)
    tree.draw()
#%%
sentence = 'The brown fox is quick and he is jumping over the lazy dog'
t = createSentenceTree(sentence)
print (t)
#%%
pt = processSentenceTree(t)
print (pt)
    
                      
    
