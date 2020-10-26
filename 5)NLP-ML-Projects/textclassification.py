# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:30:08 2020

@author: Mitran
"""

# datasetlink   =https://drive.google.com/file/d/1jl0EpkfAi1hUKW-DVm5quZSZEKneohY2/view?usp=sharing


import numpy as np
import re
import nltk
from  nltk.corpus import stopwords
import pickle as pic

from sklearn.datasets import load_files

review=load_files("txt_sentoken/")
# number of folders equals number of classes
                  
X,y=review.data,review.target
corpus=[]
for i,sent in enumerate(X):
    s=sent.lower()  
    sw=''
    if(i%100==0):
        print(i)
    
    for word in s.split():
       
        if(word not in set(stopwords.words('english')) ):
           
            new_word=re.sub(r'[^a-zA-Z]',' ',str(word))
            sw+=new_word      
            sw+=' '
    corpus.append(sw)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=0.8)# if a word appear in less than or equal to 3 docs it is removed
                                                                 # if a word appear in more than than or equal to 80 percent docs it is removed
x=vectorizer.fit_transform(corpus).toarray()


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)    


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

with open('classification.pickle','wb') as f:
    pic.dump(classifier,f)