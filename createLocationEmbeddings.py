#-----------------------------------------------------------------------------------------------------
# Description:
# Given the json file extracted from wikipedia dump (see https://radimrehurek.com/gensim/scripts/segment_wiki.html), 
# this script applies Named Entity recognition 
# on each article to extract geografical references mentioned in each of them. 
# It outputs a file with n lines (one line per entity (wikipedia article)). 
# Each line has the format: article_id	article_title	list_of_geografical_mentions
#-----------------------------------------------------------------------------------------------------

import spacy
import os
import sys
import logging
import unwiki
import re
import json
from smart_open import smart_open
from nltk.tokenize import sent_tokenize
import math
import namedEntityRecognition
import time
import multiprocessing as mp
import signal

#---------------------------------------------------------------------
# Configure log information
#---------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------
# Global and configuration values
#---------------------------------------------------------------------
wikipediaDumpJSON = './enwiki-latest-with-id.json.gz'
nProcess = 10
hashSize = 20 # Do not change
nerObj = namedEntityRecognition.ner()

#---------------------------------------------------------------------
# This function interates over the wikipedia dump (conveted to json by segment_wiki.py (from gemsin)) 
# to perform named entity recognition. 
# Does not load the entiry file in memory. One article at the time (memory friendly) 
#---------------------------------------------------------------------
def getLocationEmbeddinsFromWikipedia(processName, existingEmbeddinsIDs, startFromFileWithIndex, stopAtFile):

	print ("Starting " + processName)

	n = 0
	createNewFile = True
	for line in smart_open(wikipediaDumpJSON):

		# Load into a dictionary
		article = json.loads(line.decode('utf-8'))

		if n < startFromFileWithIndex:
			n += 1
			continue

		if n > stopAtFile:
			locationEmbeddins.close()
			break

		if int(article['articleID']) in existingEmbeddinsIDs[int(article['articleID']) % hashSize]:
			#logger.info ("[{}] Embeddings for article with title {} is already created".format(processName, article['title']))
			continue

		if createNewFile:
			locationEmbeddins = open('./outputs/locationEmbeddins_{}_{}.txt'.format(processName, str(n)), 'w')
			createNewFile = False

		logger.info("[{}]: Parsing article {}: {}".format(processName, str(n), article['title']))

		locationsInArticle = []
		for section_title, section_text in zip(article['section_titles'], article['section_texts']):

			# Remove wiki markups and HTML tags
			section_text = unwiki.loads(section_text, compress_spaces=True)
			section_text = re.sub(r'<.*?>', '', section_text)

			# Remove parethesis 
			section_text = re.sub("[($@*&?].*[$)@*&?]", "", section_text)

			# Tokenize into sentences
			senteces_in_section = sent_tokenize(section_text)

			# Perform Named entity recoginition at a sentence level:
			for sentence in senteces_in_section:
				signal.signal(signal.SIGALRM, handler)
				signal.alarm(10)
				try:
					listOfLocations = nerObj.getListOfLocationInSentece(sentence)
				except:
					continue
				signal.alarm(0)
				locationsInArticle.extend(listOfLocations)

		n += 1
		locationEmbeddins.write("{}\t{}\t{}\n".format(article['articleID'], article['title'], ";".join(locationsInArticle)))

		# Create new file every 10000 interations (just in case the script crash in the middle)
		if n % 1000 == 0:
			locationEmbeddins.close()
			createNewFile = True

	print ("Exiting " + processName)

#---------------------------------------------------------------------
# Register an handler for the timeout
#---------------------------------------------------------------------
def handler(signum, frame):
	print ("Function timeout. Sentence not analized")
	raise Exception("Timeout exception")


#---------------------------------------------------------------------
# Return a hash table of wikipedia articles (by IDs) that have already been analized. 
#---------------------------------------------------------------------
def getExistingEmbeddins():

	print ("Getting list of existing embeddings.")

	pathForOutputFiles = './outputs'
	existingEmbeddinsIDs = [[] for i in range(hashSize)]
	n = 0
	for root, dirs, files in os.walk(pathForOutputFiles):
		for fileName in files:

			if not fileName.endswith('.txt'):
				continue

			embeddinsInFile = open(os.path.join(root, fileName), 'r').readlines()
			for embedding in embeddinsInFile:

				article_id = int(embedding.split('\t')[0])
				n += 1
				index = article_id % hashSize
				existingEmbeddinsIDs[index].append(article_id)

	print (str(n) + " article where already processed.")

	return existingEmbeddinsIDs			


if __name__ == '__main__':

	existingEmbeddinsIDs = getExistingEmbeddins()
		
	processes = []
	for processID in range(nProcess): 

		startIndex = processID * 600000
		endIndex = (processID + 1) * 600000
		processName = "Process_{}_{}".format(str(startIndex), str(endIndex))

		process = mp.Process(target=getLocationEmbeddinsFromWikipedia, args=(processName, existingEmbeddinsIDs, startIndex, endIndex))
		processes.append(process)

	for process in processes:
		process.start()

	for process in processes:
		process.join()

	print ("All done!!!")





