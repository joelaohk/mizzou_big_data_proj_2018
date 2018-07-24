from pyspark import SparkContext
import nltk
from wordcloud import WordCloud

import os, csv

sc = SparkContext()

fileName = os.path.abspath(os.path.dirname(__file__))

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
listingsPickedAttr = listings.map(lambda items: (items[0], items[4:8], items[9:12], items[45:47])) 

reviews = sc.textFile(fileName + "/reviews.csv")
reviewsHeader = reviews.first()
reviews = reviews.filter(lambda line: line != reviewsHeader).mapPartitions(lambda x: csv.reader(x))


