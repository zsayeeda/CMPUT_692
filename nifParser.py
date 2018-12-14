#-----------------------------------------------------------------------------------------
# Description: This script converts the .nif file from diaNED corpus to a more friendly json format
# Command line arguments: path_to_imput_file path_to_output_file
#-----------------------------------------------------------------------------------------

import json
import re
import sys

#-----------------------------------------------------------------------------------------
# Converts each block in a .nif file to a dictionary. Returns a list of dictionaries
#-----------------------------------------------------------------------------------------
def convertBlocksToDics(pathForFile):

	doc = open(pathForFile, 'r').read()
	docBlocks = doc.split('\n\n')

	listOfTexts = []
	listOfAnnotations = []

	for block in docBlocks:

		blockDic = {}
		isText = False
		isAnnotation = False

		for line in block.split('\n'):
			
			line = line.strip().replace('\n', '')
			
			if line.startswith('nif:anchorOf'):
				anchor = re.search(r'\"(.*?)\"', line).group(1)
				blockDic.update({'anchor':anchor})
				isAnnotation = True

			elif line.startswith('<https://'):
				blockURL = re.search(r'\<(.*?)\>', line).group(1).replace('"', '')

			elif line.startswith('nif:referenceContext'):
				referenceContext = re.search(r'\<(.*?)\>', line).group(1).replace('"', '')
				blockDic.update({'referenceContext':referenceContext})

			elif line.startswith('itsrdf:taIdentRef'):
				taIdentRef = re.search(r'\<(.*?)\>', line).group(1).replace('"', '')
				blockDic.update({'taIdentRef':taIdentRef})

			elif line.startswith('nif:isString'):
				text = re.search(r'\"(.*?)\"', line).group(1)
				year = re.search(r'DCT=(\d+)', text).group(1)
				text = re.sub(r'DCT=\d+', '', text).strip()
				blockDic.update({'text':text})
				blockDic.update({'year':int(year)})
				isText = True

		if isText:
			blockDic.update({'blockURL':blockURL})
			listOfTexts.append(blockDic)
		elif isAnnotation:
			listOfAnnotations.append(blockDic)

	return listOfTexts, listOfAnnotations

#-----------------------------------------------------------------------------------------
# Given a block of text idenfied with the block URL, returns a list of annotations for that block
#-----------------------------------------------------------------------------------------
def getAnnotationsPerBlock(blockURL, listOfAnnotations):

	annotationsInBlock = []
	for annotation in listOfAnnotations:
		if annotation['referenceContext'] == blockURL:
			annotationsInBlock.append(annotation)

	return annotationsInBlock


#-----------------------------------------------------------------------------------------
if __name__ == '__main__':

	if len (sys.argv) != 3:
		print ("Incorrect usage. Please provide the path for the input and for the output file.")
		sys.exit()

	pathForInputFile = sys.argv[1]

	pathForOutputFile = sys.argv[2]
	listOfTexts, listOfAnnotations = convertBlocksToDics(pathForInputFile)
	listOfBlocks = []

	for block in listOfTexts:
		dic = {}
		dic['year'] = block['year']
		dic['annotations'] = getAnnotationsPerBlock(block['blockURL'], listOfAnnotations)
		dic['sentence'] = block['text']
		listOfBlocks.append(dic)
	
	jsonFile = open(pathForOutputFile, 'w')
	jsonFileStr = json.dump(listOfBlocks, jsonFile, indent=4)
	jsonFile.close()
