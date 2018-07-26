from pyspark import SparkContext

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

from nltk.stem.porter import PorterStemmer
from nltk.stem import *
from wordcloud import WordCloud

import os, csv, re, string

sc = SparkContext()

fileName = os.path.abspath(os.path.dirname(__file__))

stopWordsPath = fileName + "/stop-word-list.txt"
stopWordsSet= sc.textFile(stopWordsPath).collect()
STOPWORDSEN = set(stopWordsSet)

listings = sc.textFile(fileName + "/listings.csv") # retrieve listings data
listingsHeader = listings.first() # the header of the csv
listings = listings.filter(lambda line: line != listingsHeader).mapPartitions(lambda x: csv.reader(x)) # remove the header

"""
index	| selected attributes
--------|----------------------------------------------------
[0]		| listingID
[4-7]	| name, summary, space_desc, general_desc
[9-11]	| neighbourhood_desc, notes_to_visitor, transit_desc 
[45-46]	| latitude, longitude 
"""
listingsSelected = listings.map(lambda items: ((items[0],)+tuple(items[4:8])+tuple(items[9:12])+tuple(items[45:47]))) 
print listingsSelected.take(5)

reviews = sc.textFile(fileName + "/reviews.csv")
reviewsHeader = reviews.first()
reviews = reviews.filter(lambda line: line != reviewsHeader).mapPartitions(lambda x: csv.reader(x))

"""
index	| selected attributes
--------|----------------------------------------------------
[0]		| listingID
[-1]	| comment
"""
reviewsSelected = reviews.map(lambda items: (items[0], items[-1]))
print reviewsSelected.take(5)

def lowerAndRemovePunc(texts):
	PUNCTUATION = set(string.punctuation)
	splited = [text.encode("utf-8").split(" ") for text in texts].flatten()
	result = []
	for tkn in splited:
		lowerEliminateNewLineChar = tkn.lower().replace('\\n', '')
		removePunc = ''.join([charc for charc in lowerEliminateNewLineChar if not charc in PUNCTUATION])
		result.append(removePunc)
		removeStopWords = [wd for wd in result if not wd in STOPWORDSEN]

		stemmer = PorterStemmer()
		wordStemmed = [stemmer.stem(wd) for wd in removeStopWords]

	return [j.encode("utf-8") for j in stem if j]

listingsCleaned = listingsSelected.map(lambda items: (items[0], lowerAndRemovePunc(items[1:])))
listingGrouped = listingsCleaned.reduceByKey(lambda wordsArray1, wordsArray2: wordsArray1 + wordsArray2)
listingsForTFIDF = listingGrouped.flatMap(lambda (listingID, words): words).distinct().count()
listingsTF = HashingTF(listingsForTFIDF)
listingsIDF = IDF().fit(listingsTF)
listingsTFIDF = listingsIDF.transform(listingsTF)
