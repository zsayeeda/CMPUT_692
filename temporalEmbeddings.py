#--------------------------------------------------------------------------------------------------------------------
# Description: This script contains functions deal with the temporal signatures for wikipedia entities and annotated senteces
#--------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import sys
import logging
import cPickle as pickle
from scipy import spatial

#------------------------------------------------------------------------------------
# Global and configuration values
#------------------------------------------------------------------------------------
pathForTemporalSignatures = './resources/'

#---------------------------------------------------------------------
# Configure log information
#---------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------------
# Smooth a temporal signature (See Agarwal et al. 2018 for details)
#------------------------------------------------------------------------------------
def smooth_series(series, alpha, times):

	smoothed_series = [series[0]]
	for i in range(1, len(series)):
		smoothed_series.append(alpha*series[i]+(1-alpha)*smoothed_series[i-1])

	smoothed_series = smoothed_series[times:] + [0]*times

	return smoothed_series

#------------------------------------------------------------------------------------
# Apply smoothing n times to a temporal signature (See Agarwal et al. 2018 for details)
#------------------------------------------------------------------------------------
def apply_smoothing(series, alpha, times):

	for i in range(0, times):
		series = smooth_series(series, 0.2, i+1)

	return series

#------------------------------------------------------------------------------------
# Normalize a temporal signature (See Agarwal et al. 2018 for details)
#------------------------------------------------------------------------------------
def normalize_signature(signature):

	counts = signature['counts']
	indices = signature['indices']

	dates = zip(indices, counts)
	year_indices = np.arange(0001, 2051)
	year_counts = np.zeros(len(year_indices))

	for value in dates:
		if value[0] <=2050 and value[0] > 0:
			year_counts[value[0]-year_indices[0]] = value[1]

	signature = apply_smoothing(year_counts, 0.2, 2)

	return signature

#------------------------------------------------------------------------------------
# Given an article_name, this functions returns the temporal signature for 
# the related entity. The temporal signatures are given by diaNED. This function will only 
# retrieve them and convert them to a vector (so we can use them to compute cos similarity). 
# Paramether temporalSignatures is the object for the pkl file provided by diaNed
#------------------------------------------------------------------------------------
def retrieveTemporal(temporalSignatures, article_name):

	signature = None
	if article_name in temporalSignatures.keys():
		signature = normalize_signature(temporalSignatures[article_name])
	# else:
	# 	print(str.format("No Signature found for entity: %s") %(article_name))

	return signature

#---------------------------------------------------------------------
# Create a temporal embedding for a given year 
# This is useful to compute the temporal context of entity mentions in 
# (see Agarwal et al. 2018 for details)
#---------------------------------------------------------------------
def createTemporalEmbeddingForYear(year):

	temporalEmbeddingSignature = {'indices':[year], 'counts': [1]}

	return normalize_signature(temporalEmbeddingSignature)

#---------------------------------------------------------------------
# Return the cosine similarity between two vectors
# e.g., we can use to compute the similarity between and year and a entity (article)
#---------------------------------------------------------------------
def getCosineSimilarity(vecA, vecB):
	return (1-spatial.distance.cosine(vecA, vecB))


if __name__ == '__main__':

	# Load all signatures: 
	data = None
	logger.info("Loading temporal signatures...")
	with open(os.path.join(pathForTemporalSignatures, 'TempSig.pkl'), 'rb') as f:
		data = pickle.load(f)
	logger.info("Temporal signatures loaded.")

	# # Test: Similarity between George_H._W._Bush and 1990
	# similarity1 = getCosineSimilarity(retrieveTemporal(data, 'National_Guard_(France)'), createTemporalEmbeddingForYear(1790))
	# print ("Similarity between George_H._W._Bush and 1990: " + str(similarity1))

# 	# Test: Similarity between George_W._Bush and 1990
# 	similarity2 = getCosineSimilarity(retrieveTemporal(data, 'George_W._Bush'), createTemporalEmbeddingForYear(1990))
# 	print ("Similarity between George_W._Bush and 1990: " + str(similarity2))
	
	# Test: Similarity between George_W._Bush and 2001
	similarity3 = getCosineSimilarity(retrieveTemporal(data, 'George_W._Bush'), createTemporalEmbeddingForYear(1995))
	print ("Similarity between George_W._Bush and 2001: " + str(similarity3))

	# Test: Similarity between George_H._W._Bush and 2001
	similarity4 = getCosineSimilarity(retrieveTemporal(data, 'George_H._W._Bush'), createTemporalEmbeddingForYear(1995))
	print ("Similarity between George_H._W._Bush and 2001: " + str(similarity4))

	