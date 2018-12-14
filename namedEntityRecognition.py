import spacy

#---------------------------------------------------------------------------
# Class to perfom entity named recognition using spacy
#---------------------------------------------------------------------------
class ner:

	def __init__(self, model='xx_ent_wiki_sm'):
		self.model = model
		self.nlp = None

		self.loadLanguageModel()

	#---------------------------------------------------------------------
	# Load language model
	# See https://spacy.io/usage/models for library documentation
	#---------------------------------------------------------------------
	def loadLanguageModel(self):

		print ("Loading language model")
		self.nlp = spacy.load(self.model, disable=['parser'])
		print ("Language model loaded")

		return 

	#---------------------------------------------------------------------
	# Return a list of locations mentioned in a sentence using named entity recognition
	# See https://spacy.io/usage/linguistic-features#section-named-entities for library documentation
	#---------------------------------------------------------------------
	def getListOfLocationInSentece(self, sentence):

		doc = self.nlp(sentence)

		listOfLocations = []
		for ent in doc.ents:
			if ent.label_ == 'LOC':
				location = ent.text.replace('\n', '')
				# Temporary work arround to remove noise. Sometimes a single letter or unicodes codes is annotated as location
				if len(location) > 3 and (not location.startswith('\\')): 
					listOfLocations.append(location)

		return listOfLocations

