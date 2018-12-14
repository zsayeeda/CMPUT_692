#-----------------------------------------------------------------------------------------
# Description:
# This script annotate (using dbpedia spotlight) the senteces from the json file created by ' .py' 
# The output is a json file with annotations
# Command line arguments: path_to_imput_file path_to_output_file confidence
#-----------------------------------------------------------------------------------------

import json
import re
import spotlight
import sys
from pprint import pprint

#-----------------------------------------------------------------------------------------
# Annotate the text using DBpedia Spotlight
#-----------------------------------------------------------------------------------------
def annotateWithSpotligh(text, confidence):

	try:
		annotations = spotlight.annotate('http://model.dbpedia-spotlight.org/en/annotate', text, confidence=confidence, support=20)
	except:
		return None

	return annotations

#-----------------------------------------------------------------------------------------
# Get candidates from DBpedia Spotlight
#-----------------------------------------------------------------------------------------
def getCandidatesWithSpotligh(text, confidence):

	try:
		annotationsCandidates = spotlight.candidates('http://model.dbpedia-spotlight.org/en/candidates', text, confidence=confidence, support=2)
	except Exception as e:
		print ("Error: " + str(e))
	 	return None

	return annotationsCandidates

#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
	
	if len(sys.argv) != 4:
		print ("Error: Incorrect number of arguments. Please specify path for the input and the output files, and the confidence score for annotation.")
		sys.exit()

	path_to_imput_file = sys.argv[1]
	path_to_output_file = sys.argv[2]
	confidence = float(sys.argv[3])

	jsonFile = open(path_to_imput_file, 'r')
	corpus = json.load(jsonFile)

	sentencesWithAnnotation = []
	n = 0
	for item in corpus:

		print ("Annotating sentece " + str(n) + " out of " + str(len(corpus)))

		sentence = item['sentence']
		dbpediaAnnotations = annotateWithSpotligh(sentence, confidence)
		item.update({"annotations_dbpedia":dbpediaAnnotations})

		dbpediaAnnotationsCandidates = getCandidatesWithSpotligh(sentence, confidence)
		item.update({"annotations_candidates_dbpedia":dbpediaAnnotationsCandidates})
		sentencesWithAnnotation.append(item)

		n += 1

	annotatedFile = open(path_to_output_file, 'w')
	json.dump(sentencesWithAnnotation, annotatedFile, indent=4)
	annotatedFile.close()

	print ("All done!!! \n" + str(len(sentencesWithAnnotation)) + " sentences were annotated.")



