# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:28:05 2020

@author: Mitran
"""


# clustering unlabeled large volume text to groups efficiently

import pandas as pd
import re
from nltk.corpus import stopwords as sw
from nltk import word_tokenize as wt
import gensim

from nltk.stem import WordNetLemmatizer 
  
lemen = WordNetLemmatizer() 

data=pd.read_csv('ds.csv')
data['index']=data.index
print(data.shape)
data_Set=data.iloc[0:1000,:]
documents=[]
for sent in data_Set['headline_text']:
    line=sent.lower()
    line=re.sub(r'/W',' ',line)
    line=re.sub(r'/d',' ',line)
    nl=[lemen.lemmatize(word) for word in wt(line) if ((word not in sw.words('english'))and (len(word)>3))]      
    documents.append(nl)
    
dictionary = gensim.corpora.Dictionary(documents)


bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
bow_corpus[14]


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))