# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:12:16 2020

@author: Mitran
"""

import numpy as np
word2vec={}
words=[]
vectors_lst=[]
with open('glove.6B.50d.txt', encoding='utf-8') as f:
   for line in f:
       values=line.split()
       word=values[0]
       vec=np.asarray(values[1:],dtype='float32')
       word2vec[word]=vec
       words.append(word)
       vectors_lst.append(vec)
embedding = np.array(vectors_lst)
v,d=embedding.shape

from sklearn.metrics.pairwise import pairwise_distances      
def find_analogies(w1,w2,w3):
    for w in (w1,w2,w3):
        if(w not in word2vec):
            print(w ,'not in dictionary')
            return
    a=word2vec[w1]
    b=word2vec[w2]
    c=word2vec[w3]
    e=a-b+c
    metric='cosine'
    distances = pairwise_distances(e.reshape(1, d), embedding, metric=metric).reshape(v)
    print(distances)
    idxs = distances.argsort()[:4]
   
    for idx in idxs:
      word = words[idx]
      if word not in (w1, w2, w3): 
        best_word = word
        break    
    print(w1,'-',w2,'+',w3,'=',best_word)
find_analogies('king', 'man', 'woman')
       
       