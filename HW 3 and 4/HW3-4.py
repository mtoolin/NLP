# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:11:50 2019

@author: mtool
"""
#%%
import nltk
import numpy as np
from nltk.metrics import *
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#import pattern3.text.en
from pattern.en import tag
#nltk.download('wordnet')
from difflib import SequenceMatcher

from nltk.classify import NaiveBayesClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger

#%%
givenName = 'Michael'
nickName = 'Mike'

editDist = edit_distance(givenName,nickName)
pctDist = (SequenceMatcher(None, givenName, nickName).ratio())*100

print ('The edit distance from {} to {} is {}\n'.
       format(givenName,nickName,editDist))
print ('My given name {} and nickname {} are {:2.2f}% similar)\n'.
      format(givenName, nickName,pctDist))
#%%
#
# First two lines to Romeo and Juliet
#
twoSent = ("Two households, both alike in dignity, \
in fair Verona, where we lay our scene, \
from ancient grudge break to new mutiny, \
where civil blood makes civil hands unclean. \
From forth the fatal loins of these two foes \
a pair of star-cross'd lovers take their life; \
whole misadventured piteous overthrows \
do with their death bury their parents' strife.")

stopWords = set(stopwords.words('english'))   
wordTokens = nltk.word_tokenize(twoSent)   
cleanSentence = [w for w in wordTokens if not w in stopWords] 
    
#print(word_tokens) 
print(cleanSentence)

#%%
ps = nltk.stem.PorterStemmer()
ps = nltk.stem.PorterStemmer()
sno = nltk.stem.SnowballStemmer('english')
rstem = nltk.stem.RegexpStemmer('ing$|s$|ed$', min=4)
ls = nltk.stem.LancasterStemmer()
ws = nltk.wordnet.WordNetLemmatizer()

rsWords = []
psWords = []
ssWords = []
lsWords = []
wsWords = []
#
# Create a list of stemmed words from each stemmer
# and include the lemmatizer so we only need 1 For loop
#
for w in wordTokens:
    rsWords.append(rstem.stem(w))
    psWords.append(ps.stem(w))
    ssWords.append(sno.stem(w))
    lsWords.append(ls.stem(w))
    wsWords.append(ws.lemmatize(w))
#
# Visually examine each stemmer to see how many are not morphological roots
#
print('                           Stemmer Table')
print('----------------------------------------------------------------------------------')
print(' {:13} {:13} {:13} {:13} {:13} {:13}'.format('Tokens','WordNet','Regex','Porter',
      'Snowball','Lancaster'))
print('----------------------------------------------------------------------------------')
for i in range(len(rsWords)):
    print(' {:13} {:13} {:13} {:13} {:13} {:13}'.format(wordTokens[i],wsWords[i],
          rsWords[i], psWords[i], ssWords[i], lsWords[i]))
#%%
wordWrong = 2
regxWrong = 4
portWrong = 8
snobWrong = 7
lancWrong = 20

wordPerct = round(1-(wordWrong/len(wsWords)),2)*100
regxPerct = round(1-(regxWrong/len(rsWords)),2)*100
portPerct = round(1-(portWrong/len(psWords)),2)*100
snobPerct = round(1-(snobWrong/len(ssWords)),2)*100
lancPerct = round(1-(lancWrong/len(lsWords)),2)*100

stemPercList = [wordPerct, regxPerct, snobPerct, portPerct, lancPerct]
stemmers = ['WordNet', 'Regex', 'Snowball', 'Porter','Lancaster']
colors =['r','b','c','y','g']
plt.figure(figsize=(12,8))
plt.bar(stemmers, height=stemPercList, color=colors)
plt.title('Stemmer Performance',fontsize=18)
plt.xlabel('Stemmer Name', fontsize=13)
plt.ylabel('Percent', fontsize=13)
for i, v in enumerate(stemPercList):
    plt.text(i-.1 , v+.5, str(v), color='black')
plt.show()
#%%
longSentence = 'I was going to the park today and I found an old friend \
I had not seen in ten years.'
shortSentence = 'He ran the forty in 8.3 seconds'
taggedLongSent= tag(longSentence)
taggedShortSent = tag(shortSentence)

print('Original long sentence:')
print('-----------------------')
print(longSentence,'\n')
print('POS tagged sentence:')
print('--------------------')
print(taggedLongSent,'\n')

print('Original short sentence:')
print('-----------------------')
print(shortSentence,'\n')
print('POS tagged sentence:')
print('--------------------')
print(taggedShortSent,'\n')
#%%
longTokens = nltk.word_tokenize(longSentence)
shortTokens = nltk.word_tokenize(shortSentence)
newLongSentTagged = nltk.pos_tag(longTokens, tagset = 'universal')
newShortSentTagged = nltk.pos_tag(shortTokens, tagset = 'universal')

print('Original long sentence:')
print('-----------------------')
print(longSentence,'\n')
print('POS tagged sentence:')
print('--------------------')
print(newLongSentTagged,'\n')

print('Original short sentence:')
print('-----------------------')
print(shortSentence,'\n')
print('POS tagged sentence:')
print('--------------------')
print(newShortSentTagged,'\n')
#%%
newsSentence = "The measure would block the President from accessing some \
funds to construct a wall on the southern border."
newsTokens = nltk.word_tokenize(newsSentence)
newsTaggedNLTK = nltk.pos_tag(newsTokens, tagset = 'universal')
newsTagged =  tag(newsSentence)
print('Original news sentence:')
print('-----------------------')
print(newsSentence,'\n')
print('nltk POS tagged sentence:')
print('-------------------------')
print(newsTaggedNLTK,'\n')
print('Penn POS tagged sentence:')
print('-------------------------')
print(newsTagged)
