# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:30:39 2020

@author: Mitran
"""


"""
we are going to build word 2 vec with negative sampling in numpy

1) load
2) build
3)test

initialize parameters 
calculate negative sampling
calculate subsampling

loop epoch
  loop sentece
    loop through window
    
      context= sentence[mid-window...mid+window]
      sgd(middle word,context)
      
      neg_word=sample from  p(w)
      sgd(neg word,contxt)

"""
# nltk have brow corpus for large text data

import numpy as np
import matplotlib.pyplot as plt
from nltk import sent_tokenize as st
from nltk import word_tokenize as wt
from nltk.corpus import stopwords as sw
from sklearn.metrics.pairwise import pairwise_distances
import re
import string
import sys
from nltk.corpus import brown
def rem_punc(s):
    return s.translate(str.maketrans('','',string.punctuation))
def neg_sam(sen,V):
    # Pn(w) = prob of word occuring
  # we would like to sample the negative samples
  # such that words that occur more often
  # should be sampled more often
    freq=np.zeros((V))
    for ln in sen:
        for word in ln :
           freq[word]+=1
        # smoothing
    neg_sam=freq**0.75
    neg_sam=neg_sam/neg_sam.sum()
    return neg_sam
def get_context(pos, sentence, window_size):
   start = max(0, pos - window_size)
   end = min(len(sentence), pos + window_size+1)
   context=[]
   for i,w in enumerate(sentence[start:end],start=start):
       if i!=pos:
           context.append(w)
   return context

def get_text_brown():
    sent= list(brown.sents())[:15000]

    wordcount={}
    for i in sent:
        for word in i:
          if word not in ["'",       "(",       ")",     ",",     ".",     "''"]  :
            if word in wordcount:
            
              wordcount[word]+=1
            else:
                 wordcount[word]=1
    word_count=sorted(wordcount.items(),key=lambda x:x[1],reverse=True)
    word2id={w[0]:i for i,w in enumerate(list(word_count))}    
    vect_sent=[]
    for i in sent:
        s=[word2id[w] for w in i if w in word2id]
        vect_sent.append(s)
            
    return vect_sent,word2id
            
def get_text():
    file=open('raw.txt','r')
    word_count={}
    f=file.read()
    for line in st(f):
       
        if line not in '[(*-|=\{\}])':
            ln=re.sub(r'/W',' ',line)
            
            ln=ln.replace('"',' ')            
            words=rem_punc(ln).lower().split()
            if len(words) > 1:
                # if a word is not nearby another word, there won't be any context
                for word in words:
                    if(word in word_count):
                        word_count[word]+=1
                    else:
                        word_count[word]=1
                    
    print("finished counting")
    V = 20000
    V = min(V, len(word_count))
    word_count=sorted(word_count.items(),key=lambda x:x[1],reverse=True)
    word_id={w[0]:i for i,w in enumerate(list(word_count))}
   
    sents = []
   
    for line in st(f):     
       
        if line not in '[(*-|=\{\}])':
            ln=re.sub(r'/W',' ',line)            
            ln=ln.replace('"',' ')            
            words=rem_punc(ln).lower().split()
           
            if len(words) > 1:                    
                senten=[word_id[w] for w in words if w in word_id ]               
                sents.append(senten)       
    return sents,word_id

    
def sigmoid(x):     
     return 1/(1 + np.exp(-x))
  
  
  
def sgd(word,contet_words,label,lr,w1,w2): # increase the prob of label 1 bcz they are context words (words in the window )
  #else decrease the probabilit( not in the window (randomly selct if it is negative sampling))
  # w1[word] the vectors for the word from the emmbeded matrix    shape: D
# w2[:,contet_words] shape: D x N (no of context words)   
   prob=sigmoid(w1[word].dot(w2[:,contet_words]))
   
   # gradients
   gV = np.outer(w1[word], prob - label) # D x N
   gW = np.sum((prob - label)*w2[:,contet_words], axis=1) # D
   
   
   w2[:,contet_words] -= lr*gV # D x N
   w1[word] -= lr*gW # D
   
   # return cost (binary cross entropy)
   #       max prob word in context       min prob worn not in window
   cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
   return w1,w2,cost.sum()
  
  
def analogy(wo1,wo2,wo3,wo4,w_i,i_w,weight):
    W=weight
    V, D = weight.shape
    p1 = W[w_i[wo1]]
    n1 = W[w_i[wo2]]
    p2 = W[w_i[wo3]]
    n2 = W[w_i[wo4]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    # pick one that's not p1, n1, or n2
    best_idx = -1
    keep_out = [w_i[w] for w in (wo1, wo2, wo4)]
    # print("keep_out:", keep_out)
    for i in idx:
      if i not in keep_out:
        best_idx = i
        break
    # print("best_idx:", best_idx)

    print("got: %s - %s = %s - %s" % (wo1, wo2, i_w[best_idx], wo4))
    


sentence,word_id=get_text_brown()
#sentence1,word_id1=get_text()

vocab_size=len(word_id)



config={ 
         'window_size' : 5,
          'learning_rate' : 0.025,
          'final_learning_rate' : 0.0001,
          'num_negatives' :5, # number of negative samples to draw per input word
          'epochs' : 30,
          'D' : 50 # word embedding siz
        }
# learning rate decay
learning_rate_delta = (config['learning_rate'] - config['final_learning_rate']) / config['epochs']
w1=np.random.randn(vocab_size,config['D'])#ip to hidden
w2=np.random.randn(config['D'],vocab_size)#hidden to outut

p_neg = neg_sam(sentence, vocab_size)#negative sampling

# save the costs to plot them per iteration
costs = []
total_words=sum(len(s) for s in sentence)

# for subsampling each sentence
threshold = 1e-5
p_drop = 1 - np.sqrt(threshold / p_neg)

##training
epochs=config['epochs']

for i in range(epochs):
    cost=0
    counter=0
    for sen in sentence:
        sen=[w for w in sen if np.random.random() < (1 - p_drop[w])]
        if len(sen) < 2:
          continue
      #randomly order words so we don't always see samples in the same order
        random_nos = np.random.choice(len(sen),size=len(sen),replace=True)
        for p in random_nos:
             # the middle word
             word = sen[p]
             # get the positive context words/negative samples
             context_words = get_context(p, sen, config['window_size'])
             neg_word = np.random.choice(vocab_size, p=p_neg)
             targets = np.array(context_words)
             learning_rate=config['learning_rate']
              # do one iteration of stochastic gradient descent
             w1,w2,c= sgd(word, targets, 1, learning_rate, w1, w2)
             cost += c
             c = sgd(neg_word, targets, 0, learning_rate, w1,w2)
             cost += c
        counter += 1
        if counter % 100 == 0:
          sys.stdout.write("processed %s / %s\r" % (counter, len(sentence)))
          sys.stdout.flush()
    costs.append(cost)
    # update the learning rate
          
    config['learning_rate'] =config['learning_rate'] - learning_rate_delta
    

idx2word = {i:w for w, i in word_id.items()}
for We in (w1, (w1 + w2.T) / 2):# we are concatenating w1 only van be taken or avrage can be taken
  print('cross')
  analogy('study', 'studied', 'vote', 'voted', word_id, idx2word, We)
  
  
