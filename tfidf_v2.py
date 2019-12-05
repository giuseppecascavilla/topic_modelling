import nltk
from nltk import FreqDist
# nltk.download('stopwords') # run this one time
#!python -m spacy download en # one time run

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
import csv
#import json
#import codecs 

###### REPORT on data #####
import pandas_profiling
from IPython.display import display,HTML
##########################

import os
import operator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now


from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint




# IMPORT CSV

#df = pd.read_csv(open('CodingOverview.csv','rU'), sep=';', encoding='utf-8', engine='python')
df = pd.read_csv('CodingOverview_no_Data.csv', sep=',', encoding='latin-1', engine='python')

###### REPORT on data #####
report = pandas_profiling.ProfileReport(df)
report.to_file('C:/Users/giuseppec/Desktop/profile_report.html')
##########################






#df = pd.read_csv("C:/Users/giuseppec/Desktop/PYTHONTopicModelling/codingALL.csv") #import a dataset
df.head()
#print(df.head())


#######################################
#######################################
# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

# selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)

  plt.figure(figsize=(20,5))
  #plt.figure(figsize=figsize, y_pos)
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
# rotate of 90° stuff on the x axis
  plt.xticks(rotation=90)


  plt.show()
  
# Plot the name of the column u are interested in (mycase column Text)
#freq_words(df['Text'])
freq_words(df['Code'])

########################### uncomment to print result from here########
#######################################################################


# remove unwanted characters, numbers and symbols from the TABLE uwant (my case Text)
df['Text'] = df['Text'].str.replace("[^a-zA-Z#]", " ")
  

# Let’s try to remove the stopwords and short words (<2 letters) from the reviews.
from nltk.corpus import stopwords
stop_words1 = stopwords.words('english')
#top_words = stopwords.words('english')


with open('stopwords2.txt', 'r') as f:
    stopwords2 = f.readlines()   
stop_words = stop_words1 + stopwords2



# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
df['Text'] = df['Text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in df['Text']]

# make entire text lowercase
reviews = [r.lower() for r in reviews] 


#Let’s again plot the most frequent words and see if the more significant words have come out.
freq_words(reviews, 30)

####################################################################
#################second part of the script #########################
####################################################################

#To further remove noise from the text we can use lemmatization from the spaCy library. 
#It reduces any given word to its base form thereby reducing multiple forms of 
#a word to a single word.

#run the following command once from anaconda prompt:
#python -m spacy download en # one time run


nlp = spacy.load('en', disable=['parser', 'ner'])

##################################
"""### TRY BOTH type of lemmatization """
###############################
def lemmatization(texts, tags=['NOUN', 'ADJ']): # SOFT lemmatizer filter noun and adjective
#def lemmatization(texts, tags=['NOUN', 'ADJ', 'VERB', 'ADV']):#hard LEMMATIZER 
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output
  
# tokenize the reviews and then lemmatize the tokenized reviews.
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[1])

reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1]) # print lemmatized review

reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

df['reviews'] = reviews_3

###############Lemmatized#######
freq_words(df['reviews'], 30)
###################uncomment to print result from here########



####################################################################
################# THIRD PART OF THE SCRIPT##########################
####################################################################
"""#################### LDA MODEL #############################"""
####################################################################

#We will start by creating the term dictionary of our corpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(reviews_2)

#Then we will convert the list of 
#reviews (reviews_2) into a Document Term Matrix 
#using the dictionary prepared above.

doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
# The code above will take a while. 
#Please note that I have specified the number of topics as 7 
#for this model using the num_topics parameter. 
#You can specify any number of topics using the same parameter.

lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100, chunksize=100, passes=10)



#Let’s print out the topics that our LDA model has learned.
lda_model.print_topics()

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#######################################################################
""" Cluster the sentences of my dataset based on number of topics """
######################################################################


#cluster created based on the number of topics
cluster_topic0 = []
cluster_topic1 = []
cluster_topic2 = []
cluster_topic3 = []
cluster_topic4 = []
cluster_topic5 = []
cluster_topic6 = []


indice = 0

###ELSEIF in base al numero di topic che si hanno
for doc in  doc_term_matrix:
    #print(doc)
    vector = lda_model[doc]
    if len(vector) == 1:
        topics = [vector[0][1]]
    elif len(vector) == 2:
        topics = [vector[0][1], vector[1][1]]
    elif len(vector) == 3:
        topics = [vector[0][1], vector[1][1],vector[2][1]]
    elif len(vector) == 4:    
        topics = [vector[0][1], vector[1][1],vector[2][1], vector[3][1]]
    elif len(vector) == 5:    
        topics = [vector[0][1], vector[1][1],vector[2][1], vector[3][1], vector[4][1]]
    elif len(vector) == 6:    
        topics = [vector[0][1], vector[1][1],vector[2][1], vector[3][1], vector[4][1], vector[5][1]]       
    elif len(vector) == 7:    
        topics = [vector[0][1], vector[1][1],vector[2][1], vector[3][1], vector[4][1], vector[5][1], vector[6][1]]               
        
    index_max = np.argmax(topics)
    
    
#    doc_topic.append((doc, index_max)) 
    if index_max == 0:
        #cluster_topic0.append(reviews[indice])
        cluster_topic0.append((df['Code'][indice],reviews[indice]))####prende il codice in DF
    elif index_max == 1:
        #cluster_topic1.append(reviews[indice])
        cluster_topic1.append((df['Code'][indice],reviews[indice]))
    elif index_max == 2:
        #cluster_topic2.append(reviews[indice])
        cluster_topic2.append((df['Code'][indice],reviews[indice]))
    elif index_max == 3:
        #cluster_topic3.append(reviews[indice])
        cluster_topic3.append((df['Code'][indice],reviews[indice]))
    elif index_max == 4:
        #cluster_topic3.append(reviews[indice])
        cluster_topic4.append((df['Code'][indice],reviews[indice]))
    elif index_max == 5:
        #cluster_topic3.append(reviews[indice])
        cluster_topic5.append((df['Code'][indice],reviews[indice]))        
    elif index_max == 6:
        #cluster_topic3.append(reviews[indice])
        cluster_topic6.append((df['Code'][indice],reviews[indice]))                
        
        
    indice = indice + 1
    
     
 
##########print su file di tutti i topics
def stampa(nome_file, cluster):
    with open(nome_file+'.csv', 'w') as f:
        for item in cluster:
            f.write("{},{}\n".format(item[0], item[1]))

stampa('topic1', cluster_topic0) 
stampa('topic2', cluster_topic1) 
stampa('topic3', cluster_topic2) 
stampa('topic4', cluster_topic3)
stampa('topic5', cluster_topic4)
stampa('topic6', cluster_topic5)
stampa('topic7', cluster_topic6)


###################################################
"""#################Topics Visualization##############"""
###################################################

#To visualize our topics in a 2-dimensional space we will use the pyLDAvis library. 
#This visualization is interactive in nature and displays topics along with 
#the most relevant words.

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
#vis
#pyLDAvis.display(vis)
pyLDAvis.save_html(vis, 'C:/Users/giuseppec/Desktop/output.html')
#pyLDAvis.show(vis)


###############################################
###############################################
########## EVALUATE ###########################
###############################################

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=doc_term_matrix, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list, c_v


lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=doc_term_matrix, texts=reviews_2, limit=11)


######################################################
######################################################
########## EVALUATE VERSION 2 ########################
######################################################
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, 
                                                        texts=reviews_2, start=2, limit=12, step=3)



# Show graph
limit=12; start=1; step=3;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()







