#---------------------------------------------------------------------
# Compute the temporal similarity between the annotated entities and 
# document, for all documents in the dataset and for all datasets. 
# The output is saved to a json file 
#---------------------------------------------------------------------
import os
import sys
import logging
from temporalEmbeddings import retrieveTemporal, getCosineSimilarity, createTemporalEmbeddingForYear
import cPickle as pickle
import json
import numpy as np

#------------------------------------------------------------------------------------
# Global and configuration values
#------------------------------------------------------------------------------------
pathForAnnotatedDatasets = './../../timeNED/diaNED-corpus/with_dbpedia_annotations/'
pathForAnnotatedDatasetsWithTemporalSimilarities = './../../timeNED/diaNED-corpus/with_dbpedia_annotations_and_temporal_similarities/'
pathForTemporalSignatures = './../resources/TempSig.pkl'
temporalSignatures = None

#---------------------------------------------------------------------
# Configure log information
#---------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
temporalSignatures = None

#---------------------------------------------------------------------
# Load dataset to test
#---------------------------------------------------------------------
def loadDataset(pathForDataset):

	# Load dataset annotated with DBpedia Spotlight. This dataset also contains all candidate entities for annotation.
	datasetFile = open(pathForDataset, 'r')
	dataset = json.load(datasetFile)

	return dataset

#---------------------------------------------------------------------
# Load temporal signatures provided by diaNED
#---------------------------------------------------------------------
def loadTemporalSignatures():
	global temporalSignatures

	logger.info("Loading temporal signatures...")
	with open(pathForTemporalSignatures, 'rb') as f:
		temporalSignatures = pickle.load(f)
	logger.info("Temporal signatures loaded.")

#---------------------------------------------------------------------
# Add to the dataset the temporal similarity between annotated entities and 
# document
#---------------------------------------------------------------------
def addTemporalSimilarity(dataset):

	newDataset = dataset[:]
	for x in range(len(newDataset)):
		
		document = newDataset[x]
		documentYear = document['year']
		documentDBpediaAnnotations = document['annotations_dbpedia']
		
		if not documentDBpediaAnnotations:
			continue

		# for each entity annotated with DBpedia Spotlight
		newDBpediaAnnotations = [] 
		for item in documentDBpediaAnnotations:
			entityName = item['URI'].split('/')[-1]
			
			# TODO: Fix this
			try:
				temporalSimilarity = getCosineSimilarity(retrieveTemporal(temporalSignatures, entityName),createTemporalEmbeddingForYear(documentYear))
			except:
				temporalSimilarity = -1

			item.update({"temporalSimilarity":temporalSimilarity})
			newDBpediaAnnotations.append(item)

		document['annotations_dbpedia'] = newDBpediaAnnotations
		newDataset[x] = document

	return newDataset

#---------------------------------------------------------------------
#---------------------------------------------------------------------
if __name__ == '__main__':

	loadTemporalSignatures()
	
	for root, dirs, files in os.walk(pathForAnnotatedDatasets):

		for fileName in files:

			if fileName.endswith('.json'):

				logger.info("Adding tempora similarity to file {}".format(fileName))
			
				dataset = loadDataset(os.path.join(root, fileName))
				newDataset = addTemporalSimilarity(dataset)

				newFileName = fileName.replace('.json', '_with_temporal_similaties.json')
				jsonFile = open(os.path.join(pathForAnnotatedDatasetsWithTemporalSimilarities, newFileName), 'w')
				jsonFileStr = json.dump(newDataset, jsonFile, indent=4)
				jsonFile.close()







