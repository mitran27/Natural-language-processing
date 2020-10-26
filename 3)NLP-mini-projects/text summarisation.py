# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:58:46 2020

@author: Mitran
"""



# articl summariser

import bs4
import urllib.request as scrap
import re
para=scrap.urlopen('https://en.wikipedia.org/wiki/Deep_learning').read()
soup=bs4.BeautifulSoup(para,'lxml')

text=""
clean_text=""
for para in soup.find_all('p'):
    line=re.sub(r'[[0-9]*]',' ',para.text) 
    line=re.sub(r'\s+',' ',line)
    line+=' '
    text+=line    
    
    # the further text will be used for creating hostogram and the prev for creating the summary
    line=line.lower()
    line=re.sub(r'\W',' ',line)
    line=re.sub(r'\d',' ',line)      
    line=re.sub(r'\s+',' ',line)
    clean_text+=line
    
from nltk import sent_tokenize as st
from nltk import word_tokenize as wt
from nltk.corpus import stopwords as sw

sent=st(text)
word_count={w:0 for w in wt(clean_text) if w not in sw.words('english')}
for word in wt(clean_text):
    if(word not in sw.words('english')):
        word_count[word]+=1
        
# basic histogram created
    
max_weight=max(word_count.values())
weight_histo=word_count

for word in weight_histo:
    weight_histo[word]/=max_weight
# after weighted histogram scores hast to be calculated for each sentence
sent2score={}  
for s in sent:
    for word in wt(s):
        if word in weight_histo:
        # summarisation excludes lines grater than 30
         if(len(s.split(' '))<20):
           
            if s not in sent2score:
                sent2score[s]=weight_histo[word]
            else:
                sent2score[s]+=weight_histo[word]
                
def get_summary(sent2score,n):
    
    imp_sent = sorted(sent2score.items(), key=lambda x: x[1], reverse=True)
    new_lst=[]
    for i in range (0,n):
        print(imp_sent[i])
        
get_summary(sent2score,10)
        