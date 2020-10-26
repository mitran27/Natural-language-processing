# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:37:44 2020

@author: Mitran
"""


paragraph = """hi bro . what i am coming to say is today is good day.today i am learning.
                    what i am coming to say is today is good day .
                    today i met my friends .
                    friends are good.we are good boys.
                    i am a boy"""
import numpy as np
def bow(sentence):
  newlst=[]
  for i in sentence:
      for j in i.split(' '):
          newlst.append(j)
  new_Set=set(newlst)
 
  b_o_w=np.zeros((len(sentence),len(new_Set)))
  # getting index
  vectordic={w:i for i,w in enumerate(new_Set)}
  
  for line_no,i in enumerate(sentence):
      for j in i.split(' '):
          print(line_no,vectordic[j],j)
          b_o_w[line_no,vectordic[j]]+=1
  return b_o_w
          
  
  
  
import nltk
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
new_sentence=[]
lemen=WordNetLemmatizer()

sentence=nltk.sent_tokenize(paragraph)
for i in sentence:
    line=i.lower()
    line=re.sub('[^a-zA-Z]',' ',line)
    xc=[lemen.lemmatize(x) for x in line.split() if x not in(stopwords.words('english'))]
    new_sentence.append(' '.join(xc))
    
bag=bow(new_sentence)