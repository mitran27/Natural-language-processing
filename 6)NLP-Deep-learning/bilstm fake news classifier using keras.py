# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:05:05 2020

@author: Mitran
"""
# dataset kaggle fake news classifier
import pandas as pd
df=pd.read_csv('train.csv')
df=df.dropna()[0:1000]

trainX=df.drop(['label'],axis=1)
trainy=df['label']

from keras.layers import Embedding,LSTM,Dense,Bidirectional
from keras.models import Sequential



import nltk
import re
import numpy as np
from nltk.corpus import stopwords as sw
from nltk.stem.porter import PorterStemmer
from nltk import  word_tokenize as wt
ps=PorterStemmer()

def preprocessing(text):
    #convert sentence to prprocessed corpus
  corpus=[]  
  for sentence in text:
    
    s=re.sub(r'^[a-zA-Z]',' ',sentence)
    s=s.lower().split()
    words=[ps.stem(word) for word in s  if word not in sw.words('english')]
    words=' '.join(words)
    corpus.append(words)
  return corpus
def one_hot(sentence):
    vocabulary=set([word for ln in sentence for word in wt(ln)])
    
    word2id={w:i for i,w in enumerate(vocabulary)}
    one_hot_corpus=[]
    for ln in sentence:
        matrix=[0 for i in range(len(wt(ln)))]
        for i,w in enumerate(wt(ln)):
            matrix[i]=word2id[w]
        one_hot_corpus.append(matrix)
    return one_hot_corpus,vocabulary
def padding(corpus,padding,maxlen)  :
    embed_Corpus=[]
    if(padding=='pre'):
       for sent in corpus:
           x=np.zeros((maxlen))
           for i,onh in enumerate(reversed(sent)):
           
               x[i]=onh
      
           embed_Corpus.append(np.flipud(x))  
    else:
        for sent in corpus:
           x=np.zeros((maxlen))
           for i,onh in enumerate(sent):
           
               x[i]=onh
      
           embed_Corpus.append(x)
        
    return np.array(embed_Corpus)
            
        



message=trainX.copy()

#apply stemming and stop words

nw=trainX['title']

corpus=preprocessing(nw)
corpus1,voc=one_hot(corpus)


emmbedded_docs_mine=padding(corpus1,'pre',29)


embedding_dim=40

model=Sequential()
model.add(Embedding(len(voc),embedding_dim,input_length=29))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


x_final=np.array(emmbedded_docs_mine)
y_final=np.array(trainy)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.20,random_state=42)

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=32)   






