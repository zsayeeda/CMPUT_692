#---------------------------------------------------------------------
# Compute the temporal similarity between the annotated entities and 
# document, for all documents in the dataset and for all datasets. 
# The output is saved to a json file 
#---------------------------------------------------------------------
import os
import sys
import logging
import pickle
import json
import numpy as np
from namedEntityRecognition import ner
from spatialEmbeddings import getCosineSimilarity, retrieveSpatial, createSpatialEmbeddingForLocations
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from copy import deepcopy

#------------------------------------------------------------------------------------
# Global and configuration values
#------------------------------------------------------------------------------------
pathForAnnotatedDatasetsWithTemporalSimilarities = './../timeNED/diaNED-corpus/with_dbpedia_annotations_and_temporal_similarities/'
pathForAnnotatedDatasetsWithTemporalAndLocationSimilarities = './../timeNED/diaNED-corpus/with_dbpedia_annotations_and_temporal_spatial_similarities/'
pathForSpatialSignatures = './../resources/SpatialSignatures.pkl'
pathForWordEmbeddingModels = './../resources/wordEmbeddings/word2vec'
model = None
spatialSignatures = None
locationAnnotator = None


#---------------------------------------------------------------------
# Configure log information
#---------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------------------
# Load word2vec model
#---------------------------------------------------------------------------------------
def loadModel(modelName='word2vec.6B.50d.txt'):
	global model

	logger.info("Loading the word2vec model")
	model = KeyedVectors.load_word2vec_format(os.path.join(pathForWordEmbeddingModels, modelName))
	if model:
		logger.info("Model has been loaded")
	else: 
		raise Exception("Could not load word embeddings model")


#---------------------------------------------------------------------------------------
# Load spatial signatures
#---------------------------------------------------------------------------------------
def loadSignatures():
	global spatialSignatures

	# Load all signatures: 
	logger.info("Loading spatial signatures...")
	with open(pathForSpatialSignatures, 'rb') as f:
		spatialSignatures = pickle.load(f)
	logger.info("Spatial signatures loaded.")


#---------------------------------------------------------------------
# Load dataset to test
#---------------------------------------------------------------------
def loadDataset(pathForDataset):

	# Load dataset annotated with DBpedia Spotlight. This dataset also contains all candidate entities for annotation.
	datasetFile = open(pathForDataset, 'r')
	dataset = json.load(datasetFile)

	return dataset

#---------------------------------------------------------------------
# Add to the dataset the temporal similarity between annotated entities and 
# document
#---------------------------------------------------------------------
def addSpatialSimilarity(dataset):

	newDataset = deepcopy(dataset)
	for x in range(len(newDataset)):
		
		document = newDataset[x]
		sentence = document['sentence']
		documentLocations = locationAnnotator.getListOfLocationInSentece(sentence)
		documentDBpediaAnnotations = document['annotations_dbpedia']
		
		if not documentDBpediaAnnotations:
			continue

		if not documentLocations:
			continue

		# for each entity annotated with DBpedia Spotlight
		newDBpediaAnnotations = [] 
		for item in documentDBpediaAnnotations:
			entityName = item['URI'].split('/')[-1]
			
			try:
				signature = retrieveSpatial(spatialSignatures, entityName, model)
			except ValueError as e:
				spatialSimilarity = -1
				item.update({"spatialSimilarity":spatialSimilarity})
				newDBpediaAnnotations.append(item)
				continue

			try:
				documentSpatialEmbedding = createSpatialEmbeddingForLocations(documentLocations, sentence, model)
			except Exception as e:
				spatialSimilarity = -1
				item.update({"spatialSimilarity":spatialSimilarity})
				newDBpediaAnnotations.append(item)
				continue

			try:
				spatialSimilarity = getCosineSimilarity(signature, documentSpatialEmbedding)
			except Exception as e:
				spatialSimilarity = -1
				item.update({"spatialSimilarity":spatialSimilarity})
				newDBpediaAnnotations.append(item)
				continue

			item.update({"spatialSimilarity":spatialSimilarity})
			newDBpediaAnnotations.append(item)

		document['annotations_dbpedia'] = newDBpediaAnnotations
		newDataset[x] = document

	return newDataset

#---------------------------------------------------------------------
#---------------------------------------------------------------------
if __name__ == '__main__':

	loadSignatures()
	loadModel()
	locationAnnotator = ner()
	
	for root, dirs, files in os.walk(pathForAnnotatedDatasetsWithTemporalSimilarities):

		for fileName in files:

			if fileName.endswith('.json'):

				logger.info("Adding spatial similarity to file {}".format(fileName))
			
				dataset = loadDataset(os.path.join(root, fileName))
				newDataset = addSpatialSimilarity(dataset)

				newFileName = fileName.replace('_with_temporal_similaties.json', '_with_temporal_spatial_similaties.json')
				jsonFile = open(os.path.join(pathForAnnotatedDatasetsWithTemporalAndLocationSimilarities, newFileName), 'w')
				jsonFileStr = json.dump(newDataset, jsonFile, indent=4)
				jsonFile.close()

	print ("All done!!!")







