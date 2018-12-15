Detecting Errors in Entity Linking Using Temporal and Spatial Information

How to reproduce the experiments (the other of the steps is important):

1) Download the Wikipedia dump. The dump is an XML file with all Wikipedia articles. It can be downloaded from https://dumps.wikimedia.org/enwiki/

2) Run the script 'segment_wiki.py' to extract plain text out of a raw Wikipedia dump and save the output as a JSON file.
This is the same script provided by gensim, with a small modification to include the article_id in the JSON file. 
See https://radimrehurek.com/gensim/scripts/segment_wiki.html for a complete documentation of this script. 

3) Run the script 'createLocationEmbeddings.py' to read the JSON file created on step 2 and extract the mentions of geographical locations in the articles. 

This script will iterate over all Wikipedia articles, tokenize the sentences and perform Named Entity Recognition in each sentence of each article. Thus, it will take some time to run (a few days). The script will create 10 process. Each process will take care of a batch with 600000 files.

This script will generate multiple .txt files in the ./outputs folder. Each text file contains 10000 lines. Each line correspond to one wikipedia article and has the following format:
article_id    article_name    [list_of_all_geografic_locations_mentioned_in_the_article]

ATTENTION:  Before executing this step, open the script and edit the variable 'pathToWikipediaFile' with the path to the file generated on step 2. Also, make sure that you created the folder ./outputs

4) Run the script 'convertLocationEmbeddinsToSignatures.py'. This script will combine all .txt files generated on step 3 into a single .pkl  file which is a dictionary object with the following format
    {article_title:{indices:[location_mentions], counts[locations_counts]}

You need to provide the path to the root of the output folder (the one you created to save the files in step 3) and the path to the output file.

5) Download the diaNED-2 corpus (Agarwal et. al 2018): https://www.mpi-inf.mpg.de/yago-naga/dianed/

6) Run the script 'nifParser.py'. This script converts the .nif file from diaNED corpus to a more friendly JSON format. 
Command line arguments: path_to_imput_file path_to_output_file
The output file looks like this: 
{
    "sentence": sentence
    "annotations": manual annotations in the sentence. Done by Agarwal et. al (2018). 
    "year": The temporal tag used by Agarwal et. al (2018)
}

7) [Use Python 2.x] Run the script 'annotateWithDBpediaSpotlight.py'. This script will annotate the sentences from the diaNED corpus (which were converted to json on step 6) using media spotlight. It will also get a list of all matching candidate entities rather than only the top candidate.
You need to specify: path_to_the_input_file path_to_the_output_file confidence_score_for_annotation

The output file looks like this:
{
    "sentence": sentence
    "annotations": manual annotations in the sentence. Done by Agarwal et. al (2018). 
    "annotations_dbpedia": automatic annotations done by DBpedia Spotlight
    "annotations_candidates_dbpedia": list of all matching candidate entities
    "year": The temporal tag used by Agarwal et. al (2018)
}

8) Run the script 'addTemporalSimilaritiesToDataset.py'. For each document in the diaNED corpus that was annotated by dpedia spotlight on step 7, this script will compute the temporal similarity between the document and each annoated entity. Before running the script, open it and update the variables 'pathForAnnotatedDatasets' (with the path to the files created on step 7) and 'pathForAnnotatedDatasetsWithTemporalSimilarities' (where the output of this script will be saved).  

9) Run the script 'addSpatialSimilaritiesToDataset.py'. For each document in the diaNED corpus that was annotated by dbpedia spotlight on step 7, this script will compute the spatial similarity between the document and each annotated entity. Before running the script, open it and update the variables 'pathForAnnotatedDatasetsWithTemporalSimilarities' (with the path to the files created on step 8) and 'pathForAnnotatedDatasetsWithTemporalAndLocationSimilarities' (where the output of this script will be saved).

10) Use the script 'errorsDetectionDBpedia.py' to perform tests. This script will also generate a file 'goodExamples.txt' that contains examples of errors in dbpedia spotlight annotation that where correctly detected by our approach. Before running the script, open the file and update the variable 'pathForDBpediaAnnotatedDataset' with the path for the files created on step 9. 
