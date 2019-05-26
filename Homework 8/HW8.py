# -*- coding: utf-8 -*-

""""
Created on Thu Mar 21 20:34:52 2019

@author: mtool
"""
import sys; print ("Python", sys.version)
import nltk
#from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import Tree
from pattern.en import parsetree, Chunk
import string
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup 
from statistics import mean, median

import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import re
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
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
#
# Create list of individual reviews
# Then create tfidf vectors and get feature names
#
listOfMyStopwords = ['movie','film','films','movies']
listOfReviews = []
listOfMovies = []
for k2 in textDict:
    listOfReviews.append(textDict.get(k2))
    listOfMovies.append(k2)
stopwordList = nltk.corpus.stopwords.words('english')+listOfMyStopwords
#tokenizer = RegexpTokenizer(r'\w+')
#TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwordList)
TfidfVec = TfidfVectorizer(stop_words=stopwordList)
tfidfReviewMatrix = TfidfVec.fit_transform(listOfReviews)
tfidfReviewFeatureNames = TfidfVec.get_feature_names()

#%%
numOfClustersToTest = [2, 3, 4, 5, 6, 7, 8, 9 ,10]
bestCluster = 0                                 # Initalize this to zero
bestSilhoutteScore = 0
clusterListByMovie = []                                 # Use list of tuples to keep order
#
# Test which clustering size gives the densest clusters by checking 
# silhoutte scores
# Save the best clustering to analyze 
#
for nClusters in numOfClustersToTest:
    km = KMeans(n_clusters=nClusters, random_state=10,tol=.0001, max_iter=1000)
    clusterLabels = km.fit_predict(tfidfReviewMatrix)   
    silhouette_avg = metrics.silhouette_score(tfidfReviewMatrix, clusterLabels)
    sample_silhouette_values = metrics.silhouette_samples(tfidfReviewMatrix, 
                                                          clusterLabels)
    if silhouette_avg > bestSilhoutteScore:
        bestSilhoutteScore =  silhouette_avg
        bestCluster = nClusters
        bestKM = km
        clusters = km.labels_

#
# Graph and Print out the results
#
    
    print ('For {} clusters'.format(nClusters))
    print (Counter(km.labels_))
    print ('Silhoutte score:{:.6f}\n'.format(silhouette_avg))
    
print ('-'*20)
print('The best clustering was seen with {} clusters'.format(bestCluster))
#
# Build the cluster list of tuples such that (Movie,ClusterNumber)
# Use lists here rather than dictionaires because the order a dictionary
# is created is not guaranteed. 
#        
for i, k in enumerate (listOfMovies): clusterListByMovie.append((k,clusters[i]))
       
#%%
#
# Get cluster information into form for displaying
#
clusterInfo = {}
centers = bestKM.cluster_centers_.argsort()[:,::-1]
for clusterNum in range(bestCluster):
    movies = []
    clusterInfo[clusterNum] = {}
    clusterInfo[clusterNum]['clusterNum'] = clusterNum
    keyFeatures = [tfidfReviewFeatureNames[i] for i in centers[clusterNum,:10]]
#    allFeatures = [tfidfReviewFeatureNames[i] for i in centers[clusterNum,:-1]]
    clusterInfo[clusterNum]['keyFeatures'] = keyFeatures
#    clusterInfo[clusterNum]['allFeatures'] = allFeatures
    for x, y in enumerate(clusterListByMovie):
        if clusterListByMovie[x][1] == clusterNum: movies.append(y[0])

    clusterInfo[clusterNum]['movies'] = movies

#%%
for cluster_num, cluster_details in clusterInfo.items():
    print ('Cluster {} details:'.format(cluster_num))
    print ('-'*20)
    print ('Key features:', cluster_details['keyFeatures'])
    print ('Movies in this cluster:')
    print (', '.join(cluster_details['movies']))
    print ('='*40)  
#%%
#
# Use the VADER lexicon
#
sid = SentimentIntensityAnalyzer()  
def analyzeSentimentVaderLex (review, threshold = 0.1, verbose = False):
#    tfPreProcess = TfidfVec.build_preprocessor()
#    cleanReview = tfPreProcess(review)
#    scores = sid.polarity_scores(cleanReview)
    scores = sid.polarity_scores(review)
    aggScore = scores['compound']
    finalSent = 'positive' if aggScore >threshold else 'negative'
    positive = str(round(scores['pos'],2)*100)+'%'
    final = round(aggScore, 2)
    negative = str(round(scores['neg'],2)*100)+'%'
    neutral = str(round(scores['neu'],2)*100)+'%'
    sentFrame = pd.DataFrame([[finalSent, final, positive,
                               negative, neutral]],
        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                      ['Predicted Sentiment', 'Polarity Score',
                                       'Positive', 'Negative',
                                       'Neutral']], 
                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
    if verbose: print (sentFrame);
    return(scores, sentFrame)
    
# Look at a sample of Reviews
sampleOfReviews = listOfReviews[0:5]
for review in sampleOfReviews[:10]:
    print ('Review:')
    print (review)
    print ()
    analyzeSentimentVaderLex(review, threshold =0.1, verbose = True)
    print ('-'*60)
    
#%%
print('-'*80)
print('{:10} {:10} {:10} {:10} {:10} '.format('Cluster#', 'Avg', 'Min',
     'Max','Median'))
print('-'*80)
for clusterNum in range(bestCluster):
    cluster = []
    for word in clusterInfo[clusterNum]['keyFeatures']:
        scores, sentiment = analyzeSentimentVaderLex(word, threshold = 0.1, 
                                                 verbose = False)
        cluster.append(scores['compound'])
    print('{:^10d} {:^1.4f}    {:^1.4f}     {:1.4f}    {:1.4f}\n Words: {}'.format(
            clusterNum, mean(cluster), min(cluster), max(cluster), 
            median(cluster), clusterInfo[clusterNum]['keyFeatures']))
    print('-'*80)
 
        
        

    
    
    
#%%
#
#  EXTRA CREDIT!!!
#
 
# First recreate the chunks from HW 5
nounPhraseDict = {}
for k2 in textDict:
    rawText = textDict.get(k2)
    str_tokens = nltk.word_tokenize(rawText)   # not sure if we need this.
    tree = parsetree(rawText)
    print ('\n',k2)
    for sentence_tree in tree:
        for chunk in sentence_tree.chunks:
            if (chunk.type =='NP'):
                nounPhrase = [(word.string, word.type) for word in chunk.words]
                print("NP ->",nounPhrase)
                nounPhraseDict.setdefault(k2, []).append(nounPhrase)

# First we need the list of tokens

    

#%%
#
# Change tfidfReviewMatrix Sparse Matrix into a numpy matrix
#
tfpd = pd.DataFrame(tfidfReviewMatrix.todense())   
pca = PCA(n_components=2).fit(tfpd)  
tfpdPcaTransform = pca.fit_transform(tfpd)  
       
print ('\nThe vector for each Principle Component')
print ('---------------------------------------')
print('The vector for the First Principle Component is:{} \
      \nThe vector for the Second Principle Component is {}'.format(
      pca.components_[0],pca.components_[1]))

print ('\nThe variance explained by each Principle Component')
print ('-----------------------------------------------------')
print('The variance for the First Principle Component is:{} \
      \nThe variance for the Second Principle Component is {}'.format(
      pca.explained_variance_[0],pca.explained_variance_[1]))
#%%
kmeans = KMeans(n_clusters=bestCluster)
clusterLabels = kmeans.fit_predict(tfpdPcaTransform)
centroid = kmeans.cluster_centers_
labels = kmeans.labels_
print ('\nThe centroid coordinates for each group')
print ('-----------------------------------------')
print (centroid)

print ('\nThe coordinates for each review')
print ('-----------------------------------------')
#%%
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
colors = cm.jet(clusterLabels.astype(float) / bestCluster)
ax.scatter(tfpdPcaTransform[:, 0], tfpdPcaTransform[:, 1], marker='.', s=300, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
ax.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=400, edgecolor='k')


ax.scatter(centers[0], centers[1], marker='$%d$' % i, alpha=1,
           s=400, edgecolor='k')

ax.set_title("The visualization of the clustered data.", fontsize=20)
ax.set_xlabel("Feature space for the 1st feature", fontsize=20)
ax.set_ylabel("Feature space for the 2nd feature", fontsize=20)

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % bestCluster),
            fontsize=25, fontweight='bold')
        
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)

plt.show()

#%%
#
# Begin Clustering Code
#
pca = PCA(n_components=2)  
tfpdPcaTransform = pca.fit_transform(tfpd)  

linkedTfidf=linkage(tfpdPcaTransform)

labelList = listOfMovies
# explicit interface
fig = plt.figure(figsize=(10, 20))
ax = dendrogram(linkedTfidf, orientation="left", labels=labelList)
plt.tick_params(axis= 'x',   
                which='both',  
                bottom='off',
                top='off',
                labelbottom='off')
plt.tight_layout()
ax.tick_params(axis='x', which='major', labelsize=10)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()  

#%%
#
# Plot is not working because of use of Pandas
# Printing of clusterInfo is working
# Can also use AGNES and Diane from book
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)  

model = AgglomerativeClustering(n_clusters=3)

model = model.fit(tfpdPcaTransform)

plt.title("Hierarchical Clustering")
plot_dendrogram()
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


                      
    
