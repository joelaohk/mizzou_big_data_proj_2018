from pyspark import SparkContext

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import *
from wordcloud import WordCloud

import os, csv, re, string
from itertools import chain
sc = SparkContext()

fileName = os.path.abspath(os.path.dirname(__file__))

stopWordsPath = fileName + "/stop-word-list.txt"
stopWordsSet= sc.textFile(stopWordsPath).collect()
STOPWORDSEN = set(stopWordsSet)

PRINTABLE = set(string.printable)

def isPrinable(text):
	result = True
	for char in text:
		result = result and (char in PRINTABLE)
	return result

listings = sc.textFile(fileName + "/listings.csv") # retrieve listings data
listingsHeader = listings.first() # the header of the csv
listings = listings.filter(lambda line: line != listingsHeader and isPrinable(line)).mapPartitions(lambda x: csv.reader(x)) # remove the header
"""
index	| selected attributes
--------|----------------------------------------------------
[0]		| listingID
[4-7]	| name, summary, space_desc, general_desc
[9-11]	| neighbourhood_desc, notes_to_visitor, transit_desc 
[45-46]	| latitude, longitude 
"""
listingsSelected = listings.filter(lambda row: len(row) > 50).map(lambda items: ((items[0],)+tuple(items[4:8])+tuple(items[9:12])+tuple(items[45:47]))) 
#print listingsSelected.take(5)

reviews = sc.textFile(fileName + "/reviews.csv")
reviewsHeader = reviews.first()
reviews = reviews.filter(lambda line: line != reviewsHeader and isPrinable(line)).mapPartitions(lambda x: csv.reader(x))
"""
index	| selected attributes
--------|----------------------------------------------------
[0]		| listingID
[-1]	| comment
"""
reviewsSelected = reviews.filter(lambda row: len(row) > 5).map(lambda items: (items[0], items[-1]))
#print reviewsSelected.take(5)

def lowerAndRemovePunc(texts):
	PUNCTUATION = set(string.punctuation)
	splited = list(chain.from_iterable([text.split(" ") for text in texts]))
	removedPunc = []
	for tkn in splited:
		lowerEliminateNewLineChar = tkn.lower().replace('\\n', '')
		removePunc = ''.join([charc for charc in lowerEliminateNewLineChar if not charc in PUNCTUATION])
		removedPunc.append(removePunc)

	removeStopWords = [wd for wd in removedPunc if not wd in STOPWORDSEN]
	stemmer = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	wordStemmed = [lemmatizer.lemmatize(wd) if lemmatizer.lemmatize(wd).endswith("e") else stemmer.stem(wd) for wd in removeStopWords]
	return [j.encode("utf-8") for j in wordStemmed if j]

listingsCleaned = listingsSelected.map(lambda items: (items[0], lowerAndRemovePunc(items[1:8])))
listingsGrouped = listingsCleaned.reduceByKey(lambda wordsArray1, wordsArray2: wordsArray1 + wordsArray2)

listingsForHash = listingsGrouped.map(lambda (listingID, words): (listingID, words, HashingTF(len(set(words)))))
listingsHashed = listingsForHash.map(lambda (listingID, words, hasher): (listingID, zip(words, hasher.transform(words))))
listingsHashed.saveAsTextFile(fileName + "/result/listingsHashed/")

reviewsCleaned = reviewsSelected.map(lambda items: (items[0], lowerAndRemovePunc(items[1:])))
reviewsGrouped = reviewsCleaned.reduceByKey(lambda wordsArray1, wordsArray2: wordsArray1 + wordsArray2)

reviewsForHash = reviewsGrouped.map(lambda (listingsID, words): (listingsID, words, HashingTF(len(set(words)))))
reviewsHashed = reviewsForHash.map(lambda (listingsID, words, hasher): (listingsID, zip(words, hasher.transform(words))))
reviewsHashed.saveAsTextFile(fileName + "/result/reviewsHashed/")

# listingsForJoin = listingsHashed.map(lambda (listingsID, word_tf): (listingsID, ("host_desc", word_tf)))
# reviewsForJoin = reviewsHashed.map(lambda (listingsID, word_tf): (listingsID, ("reviews", word_tf)))
listingsReviews = listingsHashed.join(reviewsHashed)
listingsReviews.saveAsTextFile(fileName + "/result/joined/")

def saveImage(listingsID, listingImage, reviewImage):
	if not os.path.exists(fileName + "/result/image/"):
		os.makedirs(fileName + "/result/image/")
	listingImage.to_file(fileName + "/result/image/" + listingsID + "-hostDesc.png")
	reviewImage.to_file(fileName + "/result/image/" + listingsID + "-review.png")

def getCloud(wordsTF):
	return WordCloud().generate_from_frequencies({k: v+1 for k,v in dict(wordsTF).items()})

joinedImage = listingsReviews.map(lambda (listingsID, listing_review): (listingsID, getCloud(listing_review[0]), getCloud(listing_review[1])))
joinedImage.map(lambda (listingsID, listingImage, reviewImage): saveImage(listingsID, listingImage, reviewImage)).take(30)
