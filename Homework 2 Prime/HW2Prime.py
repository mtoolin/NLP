# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 12:17:42 2019

@author: Michael Toolin
"""

import platform; print(platform.platform())
import sys; print ("Python", sys.version)
import nltk; print("NLTK", nltk.__version__)
from bs4 import BeautifulSoup ; print("Beautiful Soup")
import requests; print("requests", requests.__version__)
nltk.download('punkt')
nltk.download('stopwords')
from urllib import request
import requests
import re; print("re", re.__version__)
import numpy as np; print("Numpy",np.__version__)
import matplotlib.pyplot as plt#; print("Matplotlib",plt.__version__)
import seaborn as sns
import sklearn; print("Scikit-Learn", sklearn.__version__)
import os
import re
import string
print (os.environ['CONDA_PREFIX'])
#%%
###########################################################
#                                                         #
#  downLoadBook                                           #
#                                                         #
# This procedure takes a url  as input                    #
# and downloads the text from the URL and saves the       #
# text locally in ./data directory                        #
#                                                         #
# Input:                                                  #
#   url - url of text to download                         #
#                                                         #
# Output:                                                 #
#   savedFilename.txt - locally stored text file          #
#                                                         #
###########################################################

def downLoadBook (url):
    response = request.urlopen(url)
    localFile = response.read().decode('utf8')
    if debug: 
        print ('Book read in...',len(localFile))
    return localFile
#%%

#%%
def createCorpusList(bookList):
    page =requests.get('http://www.gutenberg.org/wiki/Children%27s_Instructional_Books_(Bookshelf)')
    soup = BeautifulSoup(page.text,'html.parser')
    listOfURL = []
    textURL = []
#
# Get all URL's from main page for each ebook listing
# Each eBook can be downloaded in multiple formats, this URL points to 
# the listing of all the formats for each eBook
#
    for a in soup.find_all('a',href=re.compile('ebooks')):
        if debug:
            print ("Found the URL: ", a['href'])
        listOfURL.append('http:'+str(a['href']))
    print('Total of URLs found = ', len(listOfURL))
            
#
# Each eBook has multiple formats that can be downloaded and some of the
# file names do not exactly correlate to the URL, so this parses each
# eBooks listing of various formats and file names.
#
    for eachURL in listOfURL:
        eachPage = requests.get(eachURL)
        newSoup = BeautifulSoup(eachPage.text, 'html.parser')
        
        title = newSoup.find("meta", property = "og:title")
        bookTitle = title["content"]
            
        for i in newSoup.find_all('a',href=re.compile('.txt')):
                textURL = ('http:'+str(i['href']))
                bookList.append({'title': bookTitle,
                                'url': textURL,
                                'lexDiv': 0,
                                'vocabularySize' : 0,
                                'totalVocabularySize' : 0,
                                'sortedVocab':'',
                                'lexicalDiversityScore' : 0,
                                'numLongWords': 0,
                                'text': '',
                                'normalizedVocabScore': 0 ,
                                'normalizedLongWordScore': 0,
                                'textDifficulty': 0})
        
    return bookList
#%%
def normalize(myScore):
    maxValue = max(myScore)
    return [(score/maxValue) for score in myScore]
#%%
###########################################
#                                         #
# Various text/word scoring routines      #
#                                         #
###########################################

#
# Returns the Lexical Diverstiy
#
def lexicalDiversity (text):
    return len(set(text)) / len(text)

def percentage (count, total):
    return (count/total) * 100
#
# Returns the Vocabulary Size
#
def vocabularySize (text):
    return len(set(text))
#
# Return text difficulty
#
def textDifficultyScore (lexDiv, normVocab, longVocab):
    return((1/3)*lexDiv + (1/3)*normVocab) + (1/3)*longVocab

###################################################################
#                                                                 #
# longWordsStats                                                  #
# Returns the dictionary filled in with Long Words Stats          #
#   Input:                                                        #
#      text    - the text to analyze                              #
#      id      - the dictionary to fill in                        #
#      wordLen - The number of letters that defines a long word   #
#                                                                 #
#   Output: id dictionary values filled in                        #
#      numLongWords  - number of unique long words                #
#      freqLongWords - long words appearing more than wordFreq    #
#                      times                                      #
#      numFreqLngWds - number of frequent long words              #
#      lordWordLex   - lexical score of the set of long words     #
#                                                                 #
###################################################################

def longWordsStats (text, wordLen):
    
    vocab = set(text)
    longWordsList = [w for w in vocab if len(w) > wordLen]
    longWordsCount = len(longWordsList)
    if debug:
        print('Number of Long words = ', longWordsCount)
        print('Long Word Lex', ids['longWordLex'])
    return (longWordsCount)

#
# From Chapter 3 of Sarkar
#
def removeCharactersAfterTokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filteredTokens = filter(None, [pattern.sub('', token) for token in tokens])
    return (filteredTokens)
#%%
#
# Functions to clean the text
#
# List of possible start points in the text. Only one of them can be
# used, once one is found in the text, the text is cleaned to that start
# point and the end point is found.
#
def buildGutenbergStartTagList():
    INTRODUCTIONARY_START_TAG = "\r\nintroductory.\r\n\r".lower()
    CONTENTS_START_TAG = "\ncontents\r".lower()
    PREFACE_START_TAG = "^PREFACE."
    CONTENTS_CENTER_START_TAG = "contents.\r\n\r"
    CONTENTS_TILDE_START_TAG = "\n~contents~"
    CONTENTS_OF_THE_BOOKS = "\n[Contents of the Books".lower()
    CHAPITRE_PREMIER_FRENCH_START_TAG = "\nCHAPITRE PREMIER.".lower()
    TRANSCRIBER_NOTE = "Transcriber's note: There was no Table of Contents in the original book"
    LE_CONSEIL_FRENCH_START_TAG = "\nLE CONSEIL".lower()
    START_OF_THIS_PROJECT_GUTENBERG_EBOOK_STYLE_1_START_TAG ='START OF THIS PROJECT GUTENBERG EBOOK'.lower()
    START_OF_THIS_PROJECT_GUTENBERG_EBOOK_STYLE_2_START_TAG ='START OF THE PROJECT GUTENBERG EBOOK'.lower()
   

    StartTags = [PREFACE_START_TAG, CONTENTS_START_TAG, CONTENTS_CENTER_START_TAG,
                 CONTENTS_OF_THE_BOOKS, CONTENTS_TILDE_START_TAG,
                 INTRODUCTIONARY_START_TAG, CHAPITRE_PREMIER_FRENCH_START_TAG,
                 TRANSCRIBER_NOTE, LE_CONSEIL_FRENCH_START_TAG,
                 START_OF_THIS_PROJECT_GUTENBERG_EBOOK_STYLE_1_START_TAG,
                 START_OF_THIS_PROJECT_GUTENBERG_EBOOK_STYLE_2_START_TAG,
                 ]
    return (StartTags)

def removeGutenbergLicense(text, startTags):
    textWithoutLicenseStartTags = None
    textWithoutLicense = None
    
    startTagPosition, textWithoutLicenseStartTags = removeGutenbergStartTags(text, startTags)
    endTagPosition, textWithoutLicense = removeGutenbergHeaderEndTags(textWithoutLicenseStartTags)
    
    return endTagPosition, textWithoutLicense

def removeGutenbergStartTags(text, startTags):
    position = -1    
    
    for item in startTags:
        position, textWithoutLicense = removeGutenbergHeaderGeneric(text, item)
        if position > -1:
            break
    
    return position, textWithoutLicense

def removeGutenbergHeaderEndTags(text):
    END_OF_THE_PROJECT_GUTENBERG_EBOOK = "end of project gutenberg's"
    END_OF_PROJECT_GUTNEBERGS_EBOOK = "end of the project gutenberg ebook"
    endTagLocation = -1
    position = -1
    
    endTagLocation = text.find(END_OF_THE_PROJECT_GUTENBERG_EBOOK)
    
    if endTagLocation > -1:
        position = endTagLocation
        
    else:
        endTagLocation = text.find(END_OF_PROJECT_GUTNEBERGS_EBOOK)
        position = endTagLocation
    
    if position > 1:
        #remove all of the project gutenberk and tags
        text = text[:(position-1)]
    
    return position, text

def removeGutenbergHeaderGeneric(text, topSeperator):
    startTagLocation = -1
    position = -1
    contentTagLen = -1
    
    if startTagLocation > -1:
        position = startTagLocation
        contentTagLen = len(topSeperator)
        updatedPosition = startTagLocation + contentTagLen + 1
        
    else:
        position = -1
        
    if position > -1:
        text = text[updatedPosition:]
        
    return position, text
#%%

#%%
#
# Parse the text passeed in the URL field of the dictionary passed
# in and fill in the rest of the dictionary fields
#
def nlpPipeLine (d):
    html = downLoadBook(d['url'])
    rawSoup = BeautifulSoup(html, 'html.parser').get_text(strip=True).lower()
    startTags = buildGutenbergStartTagList()
    position, rawTextWithoutLicense = removeGutenbergLicense(rawSoup,startTags)
    if debug:
        print('postion',position)
        print('Length raw text', len(rawTextWithoutLicense))
#
# Tokenize the text including header information
#
    tokens = nltk.word_tokenize(rawTextWithoutLicense)
#
# Remove characters after tokenization
#
    cleanedTokens = removeCharactersAfterTokenization(tokens)
    
    text = nltk.Text(cleanedTokens)
    totalVocabularySize =  len(text)
    
#
# Normalize the words and build the vocabulary
# lower case all the alpah characters
#
    words = [w.lower() for w in text]
    
    #build the vocabulary
    vocab = sorted(set(words))
    vocabularySize = len (vocab)
    
 
    print ('Length of text = ', len(text))
    lexicalDiversityResult = lexicalDiversity(text)
    lexDiv = 1.0 * len(text) / len(set(text))
#
# Fill in the remainder of the dictionary for individual scores
#
    d['vocabularySize'] = vocabularySize
    d['totalVocabularySize'] = totalVocabularySize
    d['lexDiv'] = lexDiv
    d['text'] = words
    d['sortedVocab'] = vocab
    d['lexicalDiversityScore'] = lexicalDiversityResult
    d['numLongWords'] = longWordsStats(text, 15)

    return d
#%%
#
# Draw a 2x2 plot showing various scores for the text
#
def graphScores():
    fig =plt.figure(figsize = (12,12))
    #plt.title('Scoring 104 Gutenburg Texts', fontsize=17,)
#    ax = fig.add_subplot(1,2,1)
#    plt.hist(longLexScoreList,bins=104, facecolor='green', alpha=0.75)
#    ax.set_ylabel('Frequency', fontsize = 15)
#    ax.set_xlabel('Long Word Lexical Score',fontsize=15)
#    plt.axvline(x=np.mean(longLexScoreList), color='red', alpha=0.75, linewidth = 2)

    ax = fig.add_subplot(3,1,1)
    plt.hist(lexScoreList,bins=104, facecolor='green', alpha=0.75)
    ax.set_ylabel('Frequency', fontsize = 15)
    ax.set_xlabel('All Word Lexical Score',fontsize=15)
    plt.axvline(x=np.mean(lexScoreList), color='red',
                alpha=0.75, linewidth = 2)

    ax = fig.add_subplot(3,1,2)
    plt.hist(totalLongWrdListNorm,bins=104, facecolor='green', alpha=0.75)
    ax.set_ylabel('Frequency', fontsize = 15)
    ax.set_xlabel('Normalized Number of Long Words',fontsize=15)
    plt.axvline(x=np.mean(totalLongWrdListNorm), color='red', 
                alpha=0.75, linewidth = 2)

    ax = fig.add_subplot(3,1,3)
    plt.hist(vocabScoreListNorm,bins=104, facecolor='green', alpha=0.75)
    ax.set_ylabel('Frequency', fontsize = 15)
    ax.set_xlabel('Normalized Vocabulary Score',fontsize=15)
    plt.axvline(x=np.mean(vocabScoreListNorm), color='red', 
                alpha=0.75, linewidth = 2)

    plt.show()
#    
# Box plost of Text Difficulty
#    
    plt.boxplot(textDiff,vert=False)
    plt.title('Text Difficulty Score')
    plt.show()
#
# Draw scatter plots    
#
    fig =plt.figure(figsize = (10,12))
    ax = fig.add_subplot(2,1,1)
    plt.scatter(vocabScoreListNorm, lexScoreList)
    ax.set_ylabel('Lexical Score', fontsize = 15)
    ax.set_xlabel('Normalized Vocabulary Score',fontsize=15)

    plt.show()
    plt.close()
    
    return
#%%
#
# Main routine 
#
debug = 0
bookList = []
textDiff = []
longLexScoreList =[]
lexScoreList = []
vocabScoreList = []
totalWordList = []
totalLongWrdList = []
#
# Scrape the web page and create a list of dictionaries for each text
# Fill in each dictionaries URL for that particular text
#
createCorpusList (bookList)

#
# Iterate through list of URL's to clean and score each text
#
for ids in bookList:
    ids = nlpPipeLine(ids)

    lexScoreList.append(ids['lexicalDiversityScore'])
    vocabScoreList.append(ids['vocabularySize'])
    totalWordList.append(ids['totalVocabularySize'])
    totalLongWrdList.append(ids['numLongWords'])
    
vocabScoreListNorm = normalize(vocabScoreList)
totalLongWrdListNorm = normalize(totalLongWrdList)

#
# Fill in normalized scores into each dictionary
#
for ids, vocabScore in zip(bookList, vocabScoreListNorm):
    ids['normalizedVocabScore'] = vocabScore
    
for ids, lngWrdScore in zip(bookList, totalLongWrdListNorm):
    ids['normalizedLongWordScore'] = lngWrdScore

for d in bookList:
    d['textDifficulty'] = textDifficultyScore(d['lexDiv'],
                                              d['normalizedVocabScore'], 
                                              d['normalizedLongWordScore'])      
    textDiff.append(d['textDifficulty'])
    
graphScores()

#%%
                                
#%%

