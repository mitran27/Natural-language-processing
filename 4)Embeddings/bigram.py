# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:50:10 2020

@author: Mitran
"""


paragraph=""" Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.

Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information, e.g. in the forms of decisions. Understanding in this context means the transformation of visual images (the input of the retina) into descriptions of the world that make sense to thought processes and can elicit appropriate action. This image understanding can be seen as the disentangling of symbolic information from image data using models constructed with the aid of geometry, physics, statistics, and learning theory.

The scientific discipline of computer vision is concerned with the theory behind artificial systems that extract information from images. The image data can take many forms, such as video sequences, views from multiple cameras, multi-dimensional data from a 3D scanner or medical scanning device. The technological discipline of computer vision seeks to apply its theories and models to the construction of computer vision systems.

Sub-domains of computer vision include scene reconstruction, event detection, video tracking, object recognition, 3D pose estimation, learning, indexing, motion estimation, visual servoing, 3D scene modeling, and image restoration. Computer Vision, often abbreviated as CV, is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos.

The problem of computer vision appears simple because it is trivially solved by people, even very young children. Nevertheless, it largely remains an unsolved problem based both on the limited understanding of biological vision and because of the complexity of vision perception in a dynamic and nearly infinitely varying physical world.

In this post, you will discover a gentle introduction to the field of computer vision.

The goal of the field of computer vision and its distinctness from image processing.

What makes the problem of computer vision challenging.

START END

Typical problems or tasks pursued in computer vision.

Discover how to build models for photo classification, object detection, face recognition, and more in my new computer vision book, with 30 step-by-step tutorials and full source code

"""
import numpy as np
from nltk import word_tokenize as wt
from nltk import sent_tokenize as st
def softmax(a):
    a = a - a.max()
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)
# plot smoothed losses to reduce variability
def smoothed_loss(x, decay=0.99):
    y = np.zeros(len(x))
    last = 0
    for t in range(len(x)):
      z = decay * last + (1 - decay) * x[t]
      y[t] = z / (1 - decay ** (t + 1))
      last = z
    return y
def get_bigram_probs(sentences, V,s,e, smoothing=1):
  # structure of bigram probability matrix will be:
  # (last word, current word) --> probability
  # we will use add-1 smoothing
  # note: we'll always ignore this from the END token
   bigram_probs = np.ones((V, V)) * smoothing
   for line in st(sentences):
    sentence=wt(line)   
    for i in range(len(sentence)):
      
      if i == 0:
        # beginning word
        bigram_probs[s, word_id[sentence[i]]] += 1
      else:
        # middle word
        bigram_probs[word_id[sentence[i-1]], word_id[sentence[i]]] += 1

      # if we're at the final word
      # we update the bigram for last -> current
      # AND current -> END token
      if i == len(sentence) - 1:
        # final word
        bigram_probs[word_id[sentence[i]], e] += 1
   bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
   return bigram_probs



    



from nltk.corpus import stopwords as sw
words=[word for word in wt(paragraph)]
word_id={w:i for i,w in enumerate(set(words))}
id_word={i:w for i,w in enumerate(set(words))}
s=word_id['START']
e=word_id['END']

voc_len=len(words)

# we will also treat beginning of sentence and end of sentence as bigrams
# START -> first word
# last word -> END

bigram_probs = get_bigram_probs(paragraph, voc_len,s,e, smoothing=0.1)


 # train a logistic model
W = np.random.randn(voc_len, voc_len) / np.sqrt(voc_len)

losses = []
epochs = 1
lr = 1e-1



# what is the loss if we set W = log(bigram_probs)?
W_bigram = np.log(bigram_probs)
bigram_losses = []



























   
        
        
    