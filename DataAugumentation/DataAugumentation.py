#------------- Import Modules --------------------------#
import os
import sys
import numpy as np
import spacy
import io
from spacy import displacy
# Load the installed model "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")
import json
import urllib
import urllib.request

from SPARQLWrapper import SPARQLWrapper, JSON 

import json
import re
import string
import nltk
dictionaryWords = set(nltk.corpus.words.words())

import pandas as pd
import sklearn
from sklearn.utils import shuffle

############################################################################

# Code to get extra data from DBpedia

############################################################################

#--------------- Helper Methods Lambda ----------------------#
addSpaceBeforeUpper = lambda x: re.sub(r"(\w)([A-Z])", r"\1 \2", x)
removeWordsWithNums = lambda x: re.sub("\S*\d\S*", "", x).strip()
removeJuncChars = lambda x: re.sub('\W+',' ',x)
removePunctuations = lambda x: re.sub('['+string.punctuation+']', '', x)
toLower = lambda x: x.lower()

# Method to Get DBpedia Wikicat RDF:type Objects from the Triples
# Through Sparql Endpoint
def get_dbpedia_YagoWikicatType(dbpediaKey):

    rdf_type = []
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    
    # Sparql Query
    query = """SELECT strafter(xsd:string(?type) as ?Test,\"/Wikicat\") 
    WHERE {{ <http://dbpedia.org/resource/{}> rdf:type ?type. 
    Filter (REGEX(lcase(xsd:string(?type)),\"http://dbpedia.org/class/yago/wiki*\"))}}
    order by strlen(str(?type))
    limit 5
    """
    sparql.setQuery(query.format(dbpediaKey))  
    results = sparql.query().convert()

    if(len(results["results"]["bindings"])>0):
        for typ in results["results"]["bindings"]:
            rdf_type.append(typ["callret-0"]["value"].split("/")[-1])

    testResult = list(map(addSpaceBeforeUpper, rdf_type))
    testResult = list(map(removeWordsWithNums, testResult))
    testResult = list(map(removeJuncChars, testResult))
    
    
    return_texts = ' '.join(testResult)
    return_texts = ' '.join(list(set(return_texts.split())))

    doc = nlp(return_texts.lower())
    
    result = []
    for token in doc:
        if not token.is_stop and token.pos_ in ['NOUN'] and token.shape_ == 'xxxx' : #  ,'ADJ'
            result.append(token.lemma_)
    
    return result


# Method to get the Augumented text through DBpedia Spotlight and Sparql Query
def get_dbpedia_data(text,confidence):

    print(text)
    return_texts = []
    urlPostPrefixSpotlight = "https://api.dbpedia-spotlight.org/en/annotate"  #candidates #annotate
    args = urllib.parse.urlencode([("text", text),("confidence",confidence)]).encode("utf-8")
    request = urllib.request.Request(urlPostPrefixSpotlight, data=args, headers={"Accept": "application/json"})
    response = urllib.request.urlopen(request, timeout = 30).read()
    pydict= json.loads(response.decode('utf-8'))

    if 'Resources' in pydict:
        for item in pydict["Resources"]:

            dbpedia_key = item['@URI'].split('/')[-1]

            return_texts.append(get_dbpedia_YagoWikicatType(dbpedia_key))
    return_texts = list(set([item for sublist in return_texts for item in sublist]))
    result = text + ' ' + ' '.join(return_texts)
    
    print(result)
    return result



################################################################################################

# Augument AG News Data

################################################################################################


# Source :
#  https://skymind.ai/wiki/open-datasets
#  https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

data_ag_news = pd.read_csv("Data/AGNews/train.csv", header = None)
data_ag_news = shuffle(data_ag_news,random_state=20)
data_ag_news_new = data_ag_news.iloc[0:20000]
data_ag_news_new[0].value_counts()

texts = list(data_ag_news_new[1].values.flatten())
labels = list(data_ag_news_new[0].values.flatten())

# Get External Knowledge for AG News

counter = 0
texts_with_knowledge = []
for text in texts:
    try:
        dbpedia_text = get_dbpedia_data(text,0.7)
        texts_with_knowledge.append(dbpedia_text)
    except:
        texts_with_knowledge.append(text)
    counter += 1
    print("Current Counter : {}".format(counter))

texts_with_knowledge =list(map(toLower,texts_with_knowledge))
texts_with_knowledge =list(map(removeJuncChars,texts_with_knowledge))
texts_with_knowledge =list(map(removePunctuations,texts_with_knowledge))
texts_with_knowledge =list(map(removeWordsWithNums,texts_with_knowledge))
# texts_with_knowledge
myfile = open("Data/AGNews/agNews_with_knowledge.txt","w",encoding='utf-8')

for item in texts_with_knowledge:
    myfile.write("%s\n" % item)

myfile.close()


########################################################################################

# Augument Kaggle Dataset with External Knowledge

######################################################################################## 

# Source :

# https://www.kaggle.com/rmisra/news-category-dataset/home

# Read the data from json file
textData = []
for line in open('Data/Kaggle/News_Category_Dataset_v2.json', 'r'):
    textData.append(json.loads(line))

texts_kaggle = []
labels_tag_kaggle = []  # list of label ids
for item in textData:
    labels_tag_kaggle.append(item['category'])
    texts_kaggle.append(item['headline'])


print(len(labels_tag_kaggle))
print(len(texts_kaggle))

category = {}
labels_kaggle =[]

unique_label_kaggle = list(set(labels_tag_kaggle))

for i in range(len(unique_label_kaggle)):
    category[unique_label_kaggle[i]] = i

# Get the Labels Data
labels_kaggle = [category[item] for item in labels_tag_kaggle]

len(labels_kaggle)

# Crete a dataframe with text, labels and label tags
dataframe_kaggle = pd.DataFrame({'Data': texts_kaggle,
                                 'Label': labels_kaggle,
                                 'LabelTag':labels_tag_kaggle
                                
                                })

# Get only the specific tags to reduce the dataset
dataframe_reduced_kaggle =dataframe_kaggle[
                                            (dataframe_kaggle.LabelTag == "ENTERTAINMENT") |  
                                            (dataframe_kaggle.LabelTag == "SPORTS") |
                                            (dataframe_kaggle.LabelTag == "FOOD & DRINK") |  
                                            (dataframe_kaggle.LabelTag == "TRAVEL") |
                                            (dataframe_kaggle.LabelTag == "THE WORLDPOST") |  
                                            (dataframe_kaggle.LabelTag == "MEDIA")]



dataframe_reduced_kaggle = dataframe_reduced_kaggle.groupby('LabelTag').head(4000)
dataframe_reduced_kaggle = sklearn.utils.shuffle(dataframe_reduced_kaggle)
dataframe_reduced_kaggle.groupby('LabelTag').size().sort_values(ascending=False)

# Get the data and labels
texts_kaggle = list(dataframe_reduced_kaggle["Data"].values.flatten())
labels_kaggle = list(dataframe_reduced_kaggle["Label"].values.flatten())


# Create the augumented data from DBpedia
counter = 0
texts_with_knowledge_kaggle = []
for text in texts_kaggle:
    try:
        dbpedia_text = get_dbpedia_data(text,0.7)
        texts_with_knowledge_kaggle.append(dbpedia_text)
    except:
        texts_with_knowledge_kaggle.append(text)
    counter += 1
    print("Current Counter : {}".format(counter))

texts_with_knowledge_kaggle =list(map(toLower,texts_with_knowledge_kaggle))
texts_with_knowledge_kaggle =list(map(removeJuncChars,texts_with_knowledge_kaggle))
texts_with_knowledge_kaggle =list(map(removePunctuations,texts_with_knowledge_kaggle))
texts_with_knowledge_kaggle =list(map(removeWordsWithNums,texts_with_knowledge_kaggle))
# texts_with_knowledge
myfile = open("Data/Kaggle/kaggle_with_knowledge_20k.txt","w",encoding='utf-8')

for item in texts_with_knowledge_kaggle:
    myfile.write("%s\n" % item)

myfile.close()


#########################################################################################

# Yahoo Dataset Buildup

#########################################################################################

# Build the Yahoo Data from the Yahoo dataset

# Source 
# https://cogcomp.seas.upenn.edu/page/resource_view/89

# Data and Code to read the data has been taken from :
# https://github.com/irisliucy/Short-text-Classification/ 

Classes = 10
TEXT_DATA_DIR = 'Data/Yahoo/yahoo_' + str(Classes)

print('Processing text dataset')
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                texts.append(f.read())
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# Create the Augumented Data for Yahoo
counter = 0
texts_with_knowledge_yahoo = []
for text in texts:
    try:
        dbpedia_text = get_dbpedia_data(text,0.7)
        texts_with_knowledge_yahoo.append(dbpedia_text)
    except:
        texts_with_knowledge_yahoo.append(text)
    counter += 1
    print("Current Counter : {}".format(counter))

myfile = open("Data/Yahoo/yahoo_know.txt","w",encoding='utf-8')

for item in texts_with_knowledge_yahoo:
    myfile.write("%s\n" % item)
    
myfile.close()

