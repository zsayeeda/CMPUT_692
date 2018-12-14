#--------------------------------------------------------------------------------------------------------------------
# Description: This script contains functions deal with the spatial signatures for wikipedia entities and annotated senteces
#--------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import sys
import logging
import pickle
from scipy import spatial
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from namedEntityRecognition import ner

#---------------------------------------------------------------------
# Configure log information
#---------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------------
# Return the weighted average of the word embeddings of the locations mentioned in the article
#------------------------------------------------------------------------------------
def getAverageEmbeddingForLocations(locations, counts, model):

	locationEmbeddings = [] 
	locationEmbeddingsWeights = []
	for x in range(len(locations)):

		location = locations[x].split()

		# Getting the word embedding for one location mentioned in the article
		embeddings = []
		for word in location:
			word = word.lower()
			try:
				embeddings.append(model[word])
			except KeyError:
				continue
		embeddingForLocation = np.average(embeddings, axis=0) 
		if embeddingForLocation.shape:
			locationEmbeddings.append(embeddingForLocation)
			locationEmbeddingsWeights.append(counts[x])

	spatialSignature = [] 
	if locationEmbeddings:
		spatialSignature = np.average(locationEmbeddings, axis=0, weights=locationEmbeddingsWeights)

	return spatialSignature

#------------------------------------------------------------------------------------
# Return a embedding that represent all locations mentioned in a document
#------------------------------------------------------------------------------------
def createSpatialEmbeddingForLocations(documentLocations, sentence, model):

	counts = []
	for location in documentLocations:
		for word in location:
			count = sentence.lower().split().count(word)
			if count == 0:
				count = 1
		counts.append(count)

	return getAverageEmbeddingForLocations(documentLocations, counts, model)

#------------------------------------------------------------------------------------
# Given an article_name, this functions returns the spatial signature for 
# the related entity. The spatial signatures are the weighted average of the word embeddings
# of the locations mentioned in the article
#------------------------------------------------------------------------------------
def retrieveSpatial(spatialSignatures, article_name, model):

	signature = None
	if article_name in spatialSignatures.keys():
		article = spatialSignatures[article_name]
		signature = getAverageEmbeddingForLocations(article['indices'], article['counts'], model)
	else: 
		raise ValueError("Spatial signature for entity %s does not exist." % article_name)

	return signature


#---------------------------------------------------------------------
# Return the cosine similarity between two vectors
#---------------------------------------------------------------------
def getCosineSimilarity(vecA, vecB):
	return (1-spatial.distance.cosine(vecA, vecB))


if __name__ == '__main__':
	
	pathForWordEmbeddingModels = './../resources/wordEmbeddings/word2vec/word2vec.6B.50d.txt'
	logger.info("Loading the word2vec model")
	model = KeyedVectors.load_word2vec_format(pathForWordEmbeddingModels)
	logger.info("Model has been loaded")	


	# result = model.similar_by_word("poland", topn=100)
	# print (result)

	pathForSpatialSignatures = './../resources/spatialSignatures.pkl'
	spatialSignatures = None
	logger.info("Loading spatial signatures...")
	with open(pathForSpatialSignatures, 'rb') as f:
		spatialSignatures = pickle.load(f)
	logger.info("Spatial signatures loaded.")


	#signature = retrieveSpatial(spatialSignatures, 'George_H._W._Bush', model) #0.7247484507650677
	#signature = retrieveSpatial(spatialSignatures, 'George_W._Bush', model) #0.6428822338879983
	#sentence = 'Memo Ties Bush to Intelligence Agency Washington, D.C.'

	signature = retrieveSpatial(spatialSignatures, 'National_Guard_(France)', model)
	sentence = 'An attempt at a counter-revolution in France is put down by the National Guard at Lyons.'
	
	locationAnnotator = ner()
	documentLocations = locationAnnotator.getListOfLocationInSentece(sentence)
	documentSpatialEmbedding = createSpatialEmbeddingForLocations(documentLocations, sentence, model)


	print (signature)
	print (documentSpatialEmbedding)
	spatialSimilarity = getCosineSimilarity(signature, documentSpatialEmbedding)
	print (spatialSimilarity)




