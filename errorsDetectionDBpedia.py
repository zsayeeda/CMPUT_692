import logging
import json
import numpy as np
from prettytable import PrettyTable
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import pickle

#------------------------------------------------------------------------------------
# Global and configuration values
#------------------------------------------------------------------------------------
pathForDBpediaAnnotatedDataset = './../timeNED/diaNED-corpus/with_dbpedia_annotations_and_temporal_spatial_similarities/'
incorrectAnnotations = None
incorrectAnnotationsAux = None
correctAnnotationsAux = None
correctAnnotations = None
temporalSignatures = None

#---------------------------------------------------------------------
# Configure log information
#---------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------
# Load dataset to test
#---------------------------------------------------------------------
def loadDataset(datasetName):

	# Load dataset annotated with DBpedia Spotlight. This dataset also contains all candidate entities for annotation.
	datasetFile = open(os.path.join(pathForDBpediaAnnotatedDataset, datasetName), 'r')
	dataset = json.load(datasetFile)

	return dataset

#---------------------------------------------------------------------
# Return the accuracy of the disambiguation of the annotations made automatically
# Uses the manual annotations as ground truth
#---------------------------------------------------------------------
def computePerformance(dataset):
	global incorrectAnnotations, correctAnnotations, incorrectAnnotationsAux, correctAnnotationsAux

	disambiguationCorrect = 0
	disambiguationIncorrect = 0

	createList = False
	if not incorrectAnnotations:
		createList = True
		incorrectAnnotations = []
		correctAnnotations = []
		incorrectAnnotationsAux = [] 
		correctAnnotationsAux = [] 

	for document in dataset:

		manualAnnotations = document['annotations']
		automaticAnnotations = document['annotations_dbpedia']

		if not automaticAnnotations:
			automaticAnnotations = []

		# for each entity annotated manually
		manuallyAnnotatedEntities = []
		manuallyAnnotatedTerms = []
		for item in manualAnnotations:
			entityName = item['taIdentRef'].split('/')[-1]
			surfaceForm = item['anchor'].replace(' ', '_').strip().lower()
			manuallyAnnotatedEntities.append(entityName)
			manuallyAnnotatedTerms.append(surfaceForm)

		# for each entity annotated automatically
		automaticallyAnnotatedEntities = []
		automaticallyAnnotatedTerms = []
		for item in automaticAnnotations:
			entityName = item['URI'].split('/')[-1]
			surfaceForm = item['surfaceForm'].replace(' ', '_').strip().lower()
			automaticallyAnnotatedEntities.append(entityName)
			automaticallyAnnotatedTerms.append(surfaceForm)

		for x in range(len(automaticallyAnnotatedTerms)):
			
			term = automaticallyAnnotatedTerms[x]
			
			if term in manuallyAnnotatedTerms:
				
				index = manuallyAnnotatedTerms.index(term)
				if automaticallyAnnotatedEntities[x] == manuallyAnnotatedEntities[index]:
					disambiguationCorrect += 1
					if createList:
						auxDic = {}
						auxDic.update({"surfaceForm": term})
						auxDic.update({"entityName": automaticallyAnnotatedEntities[x]})
						auxDic.update({"sentence": document['sentence']})
						correctAnnotationsAux.append(manuallyAnnotatedEntities[index])
						correctAnnotations.append(auxDic)
				else:
					disambiguationIncorrect += 1
					if createList:
						auxDic = {}
						auxDic.update({"surfaceForm": term})
						auxDic.update({"entityName": automaticallyAnnotatedEntities[x]})
						auxDic.update({"sentence": document['sentence']})
						incorrectAnnotationsAux.append(manuallyAnnotatedEntities[index])
						incorrectAnnotations.append(auxDic)

	try:
		accuracy = float(disambiguationCorrect) / float (disambiguationIncorrect + disambiguationCorrect)
	except ZeroDivisionError:
		accuracy = -1

	return accuracy
	

#---------------------------------------------------------------------
# Remove annotations made by dbpedia if fs is bellow a given threshold. 
# Where:
# fs = {(c * gama) + (alpha * t) + (beta * s))}/(gama + alpha + beta)
# where t and s are temporal and spatial similarities between the annotated 
# sentence and annotated entities. c is the confidence score of the annotation
# provided by dbpedia spotlight
# Alpha, beta and gama are paramether to be ajusted
#---------------------------------------------------------------------
def removeWrongAnnotations(threshold, dataset, alpha, beta, gama):

	nAnnotations = 0 # Number of annotations made by dbpedia
	idkRemovals = 0 # Number of times an annotation that I don't know if it correct or not is removed
	correctRemovals = 0 # Number of times a correct annotation is correctly removed
	incorrectRemovals = 0 # Number of times a correct annotation is removed by mistake
	missingDetection = 0 # Number of times the method fails in indentify an error in dbpedia annotation
	goodExamples = [] # If the annotation is incorrect and it is removed, save it. It is usefull to show that the technique works. 
	badExamples = [] 

	# Copy the values of original alpha and beta. 
	# This is important because in some cases (in case of errors) we change these weights to zero for a specific document, 
	# but we want to use the original values for the remaining documents in the dataset
	alphaValue = alpha
	betaValue = beta

	# A copy of the dataset. We don't want to modify the original one because we gonna use it again in the next test 
	# (with different threshold, alpha, beta and gama values)
	newDataset = deepcopy(dataset) 


	# For each document in the dataset... 
	for x in range(len(newDataset)):
		
		document = newDataset[x]
		documentYear = document['year']
		documentDBpediaAnnotations = document['annotations_dbpedia']
		
		# If the document has no annotations, there is nothing to be done. Continue
		if not documentDBpediaAnnotations:
			continue

		dbpediaAnnoationsCleaned = [] # This will store the annotations that have final score above the threshold 
		# and, therefore, should not be removed. 

		# For each entity annotated with DBpedia Spotlight, check if 
		# the annotation make sense using temporal and spatial similarity and 
		# the similarity score provided by dbpedia
		for item in documentDBpediaAnnotations:
			
			alphaValue = alpha
			betaValue = beta
			nAnnotations += 1
			
			entityName = item['URI'].split('/')[-1]
			similarityScore = item['similarityScore']
			temporalSimilarity = item['temporalSimilarity']
			surfaceForm = item['surfaceForm'].replace(' ', '_').strip().lower()

			
			# Check if the spatial similarity was computed for that annotation. 
			# The spatial simiality will not be computed if the document has no mentions to any locations
			# OR if no spatial signature was found for the annotated entity 
			if 'spatialSimilarity' in item.keys():
				spatialSimilarity = item['spatialSimilarity']
			else:
				spatialSimilarity = -1


			# If either temporal similarity of spatial similarity, assign its weight to Zero
			# (so the missing value does not impact the score)
			if temporalSimilarity == -1:
				alphaValue = 0.0
			if spatialSimilarity == -1:
				betaValue = 0.0

			
			# This dictionary is used to compare the annotation made by dbpedia with the annotation
			# for the same entity that was made by the human annotator
			auxDic = {}
			auxDic.update({"surfaceForm": surfaceForm})
			auxDic.update({"entityName": entityName})
			auxDic.update({"sentence": document['sentence']})

			# final score calculation
			fs = ((alphaValue*temporalSimilarity) + (gama * similarityScore) + (betaValue*spatialSimilarity))/(gama + alpha + beta)
			
			# If the final score for the annotation is above the threshold, consider the annotation correct and keep it. 
			if fs >= threshold:
				dbpediaAnnoationsCleaned.append(item)

				# If the final score is above the threshold but the annotation is correct, 
				# it means that the technique fail to identify an error with a dbpedia annotations. 
				# We want to keep track of how many times that happens
				if auxDic in incorrectAnnotations:
					missingDetection += 1

			# If the final score for the annotation is bellow the threshold, it is considered an incorrect annotation. So do not keep it. 
			else:

				# If the annotation that we are not keeping is actually correct, it means that or method is removing correct annotations
				# We want to keep track of how many times that happens to evaluate the method. 
				if auxDic in correctAnnotations:
					incorrectRemovals += 1
					index = correctAnnotations.index(auxDic)
					correctAnnotation = correctAnnotationsAux[index]
					auxDic.update({"correctAnnotation":correctAnnotation})
					auxDic.update({"temporalSimilarity":temporalSimilarity})
					auxDic.update({"spatialSimilarity":spatialSimilarity})
					auxDic.update({"similarityScore":similarityScore})
					auxDic.update({"documentYear":documentYear})
					badExamples.append(auxDic)

				# If the annotation is in fact incorrec, the method succeeds in detecting it. 
				# Besides keep tracking of how many times that happens, we save the sentence and the annotations, 
				# so we can show that the method works. 
				elif auxDic in incorrectAnnotations:
					correctRemovals += 1
					index = incorrectAnnotations.index(auxDic)
					correctAnnotation = incorrectAnnotationsAux[index]
					auxDic.update({"correctAnnotation":correctAnnotation})
					auxDic.update({"temporalSimilarity":temporalSimilarity})
					auxDic.update({"spatialSimilarity":spatialSimilarity})
					auxDic.update({"similarityScore":similarityScore})
					auxDic.update({"documentYear":documentYear})
					goodExamples.append(auxDic)
				else:
					idkRemovals += 1

		document['annotations_dbpedia'] = dbpediaAnnoationsCleaned
		newDataset[x] = document

	return newDataset, nAnnotations, idkRemovals, correctRemovals, incorrectRemovals, missingDetection, goodExamples, badExamples

#---------------------------------------------------------------------
# Run all tests for temporal and spatial similarity
#---------------------------------------------------------------------
def runTests(dataset):

	# Those ranges were defined empirically (by testing multiple options)
	testThresholds = np.arange(0.05, 1.0, 0.05)
	testAlphas = np.arange(0.1, 0.7, 0.05)
	testBetas = np.arange(0.1, 0.7, 0.05)
	testGamas = np.arange(0.7, 1.05, 0.05)
	# testGamas = [1.0]

	# # # By-pass range and test with a single value previously defined to speed up (commend when doing serious tests)
	# 	# Best for hn
	# testAlphas = [0.25] 
	# testBetas = [0.45] 
	# testGamas = [0.75] 

		# Best for nyt
	# Best alpha: 0.30000000000000004
	# Best beta: 0.25000000000000006
	# Best gama: 1.0
	# Best threshold: 0.8000000000000002



	auxListAverage = []
	nAnnotations = 0
	printRow = [] 
	for threshold in testThresholds:
		for alpha in testAlphas:
			for beta in testBetas:
				for gama in testGamas:
					newDataset, nAnnotations, idkRemovals, correctRemovals, incorrectRemovals, missingDetection, goodExamples, badExamples = removeWrongAnnotations(threshold, dataset, alpha, beta, gama)
					accuracy = computePerformance(newDataset)
					printRow.append(["{0:0.4f}".format(threshold), "{0:0.4f}".format(alpha), "{0:0.4f}".format(beta), "{0:0.4f}".format(gama), nAnnotations, correctRemovals, incorrectRemovals, missingDetection, "{0:0.8f}".format(accuracy)])
					auxListAverage.append([threshold, alpha, beta, gama, nAnnotations, idkRemovals, correctRemovals, incorrectRemovals, missingDetection, goodExamples, badExamples, accuracy])

	pickle.dump(auxListAverage, open('hn_10.pkl', "wb" ) )

	print ("Using average beween temporal similarity and confidence score: \n")
	t = PrettyTable(['Threshold', 'alpha', 'beta', 'gama', 'nAnnotations', 'correctRemovals', 'incorrectRemovals', 'missingDetection', 'Accuracy'])
	for item in printRow:
		t.add_row(item)
	print (t)

	accuracies = list(map(list, zip(*auxListAverage)))[-1]
	thresholds = list(map(list, zip(*auxListAverage)))[0]
	alphas = list(map(list, zip(*auxListAverage)))[1]
	betas = list(map(list, zip(*auxListAverage)))[2]
	gamas = list(map(list, zip(*auxListAverage)))[3]

	# Get the highest possible alpha and beta that leads to the maximum accuracy
	highestAccuracy = max(accuracies)
	#indices = [i for i, x in enumerate(accuracies) if x == highestAccuracy]
	x = accuracies.index(highestAccuracy)
	bestAlpha = alphas[x]
	bestBeta = betas[x]
	bestGama = gamas[x]
	bestThreshold = thresholds[x]

	print ("Max accuracy: %s" % highestAccuracy)
	print ("Best alpha: %s" % bestAlpha)
	print ("Best beta: %s" % bestBeta)
	print ("Best gama: %s" % bestGama)
	print ("Best threshold: %s" % bestThreshold)

	accuracies = []
	thresholds = []
	betas = []
	alphas = []
	correctRemovals = None
	incorrectRemovals = None
	idkRemovals = None
	missingDetections = None
	goodExamples = None
	for item in auxListAverage:
		if item[1] == bestAlpha and item[2] == bestBeta and item[3] == bestGama:
			accuracies.append(item[-1])
			thresholds.append(item[0])
			if item[0] == bestThreshold:
				goodExamples = item[-3]
				badExamples = item[-2]
				idkRemovals = item[5] 
				correctRemovals = item[6]
				incorrectRemovals = item[7]
				missingDetections = item[8]


	examplesFiles = open('goodExamples.txt', 'w')
	for item in goodExamples:
		examplesFiles.write("Sentence: %s\n" % item['sentence'])
		examplesFiles.write("Year: %s\n" % item['documentYear'])
		examplesFiles.write("Surface form: %s\n" % item['surfaceForm'].replace('_', ' '))
		examplesFiles.write("DBpedia spotlight linked to: %s\n" % item['entityName'])
		examplesFiles.write("Humana annotator linked to: %s\n" % item['correctAnnotation'])
		examplesFiles.write("Temporal similarity: %s\n" % item['temporalSimilarity'])
		examplesFiles.write("Spatial similarity: %s\n" % item['spatialSimilarity'])
		examplesFiles.write("Similarity score: %s\n" % item['similarityScore'])
		examplesFiles.write('\n\n')
	examplesFiles.close()

	examplesFiles = open('badExamples.txt', 'w')
	for item in badExamples:
		examplesFiles.write("Sentence: %s\n" % item['sentence'])
		examplesFiles.write("Year: %s\n" % item['documentYear'])
		examplesFiles.write("Surface form: %s\n" % item['surfaceForm'].replace('_', ' '))
		examplesFiles.write("DBpedia spotlight linked to: %s\n" % item['entityName'])
		examplesFiles.write("Humana annotator linked to: %s\n" % item['correctAnnotation'])
		examplesFiles.write("Temporal similarity: %s\n" % item['temporalSimilarity'])
		examplesFiles.write("Spatial similarity: %s\n" % item['spatialSimilarity'])
		examplesFiles.write("Similarity score: %s\n" % item['similarityScore'])
		examplesFiles.write('\n\n')
	examplesFiles.close()

	pMissingAnnotations = float(missingDetections)/float(len(incorrectAnnotations))
	pIncorrectRemovals = float(incorrectRemovals)/float(incorrectRemovals + correctRemovals + idkRemovals)*100.0
	pCorrectRemovals = float(correctRemovals)/float(incorrectRemovals + correctRemovals + idkRemovals)*100.0
	print ("Percentage of incorrect removals: %s" % pIncorrectRemovals)
	print ("Percentage of correct removals: %s" % pCorrectRemovals)
	print ("Percentage of of incorrect annotations that were not removed: %s" % pMissingAnnotations)

	# --------------------------------------
	# Uncomment/comment lines for a 2D plot considering the best possible alpha
	# --------------------------------------
	plt.plot(thresholds, accuracies)
	plt.ylabel("Accuracy")
	plt.xlabel("Threshold")
	plt.show()
		
#---------------------------------------------------------------------
#---------------------------------------------------------------------
if __name__ == '__main__':

	datasetFileName = 'nyt-random_dbpedia_annotated_010_with_temporal_spatial_similaties.json'
	
	print ("Tests for file " + datasetFileName)
	dataset = loadDataset(datasetFileName)
	computePerformance(dataset) # This is only usefull to create the list of correct and incorrect annotations
	runTests(dataset)

	
