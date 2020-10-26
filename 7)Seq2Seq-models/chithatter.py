# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:17:04 2020

@author: Mitran
"""

import pandas as pd
import numpy as np
import re

def prepare_sentence(sent):
    x=sent
    x=x.lower()
    x=re.sub("'", "", x)
    x=re.sub('\d','',x)
    x=re.sub(r'[<>@#!.]',r' ',x)
    x= re.sub(r"([?.!,¿])", r" \1 ", x)
    x=' '.join(x.split())
    x='stxaxrt '+x+' exnxd'
    return x
def create_dataset(dataset,no_samples,source_col='source',target_col='target'):
    source=[]
    target=[]
    for i in range(no_samples):
        
        s=dataset.iloc[i][source_col] # ith sample 1st colums
        t=dataset.iloc[i][target_col]
        source.append(prepare_sentence(s))
        target.append(prepare_sentence(t))
    return source,target

from keras.preprocessing.text import Tokenizer
def avg(x):
    return (sum(x)/len(x))
def pad_sequences(text,voc_size,padding):
    max_len=max([len(i) for i in text])
    pad_Seq=[]
    if(padding=='post'):
       for sent in text:
           s=[0 for i in range(max_len)]
           s[:len(sent)]=sent
           pad_Seq.append(s)
    else :
        for sent in text:
           s=[0 for i in range(max_len)]
           s[:len(sent)]=sent[::-1]
           pad_Seq.append(s[::-1])
        
    return np.array((pad_Seq)),voc_size+1
def convert(source_index_word,target_index_word,source,prediction):
    for sents,sentt in zip(source,prediction):
        sentence_source=''
        sentence_target=''
        for word_s in sents:
            if(word_s==0):
                sentence_source +=' '
            else:
               sentence_source += source_index_word[word_s] + ' '
           
        for word_t in sentt:
            
            if(word_t==0):
                sentence_target+=' '
            
            else :               
               sentence_target += target_index_word[word_t] + ' '
    
        print('source:  ',sentence_source,'\n\n','target  :',sentence_target,'\n\n\n')
            
         
class counter():# count the ocurences of words
    def count_words(self,corpus,word_limit,clean=False):
        self.Tokens=[]
        counter={}
        for sent in corpus:
           if(clean):
               sent=self.clean(sent)
           text=sent.split() # take the required number of words
           self.Tokens.append(text)
           
           for w in text:
               if(w in counter):
                   counter[w]+=1
               else:
                   counter[w]=1
        self.C=counter
    
    def clean(string):
       stop_words=[]
       string = string.lower()
       string = re.sub(r'[^\w\s]', '', string) 
       string = ' '.join([word for word in string.split() if word not in stop_words])
       return string
    def arrange(self,o):
        return sorted(self.C.items(), key=lambda x: x[1],reverse=o)
    def most_common(self):
        return self.arrange(True)
    def least_common(self):
        return self.arrange(False)
  
def Tokenise(corpus,word_limit=100000,clean=False):
        count=counter()
        count.count_words(corpus, word_limit,clean=clean)
        
        w2i= {w:i for i,(w,c) in enumerate(count.most_common(),start=1)}
        i2w= {i:w for  w,i in w2i.items()}
        V=len(w2i)
        vec_tokens=[]
        for sent in count.Tokens:
            sent_token=[]
            for i in sent :
                sent_token.append(w2i[i])
            vec_tokens.append(sent_token)
            
        return w2i,i2w,vec_tokens,V

from keras.losses import sparse_categorical_crossentropy as scc


def loss_function(real, pred):
 
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # what mask does is  if( real is 0 it is zero else 1 )
  loss_ = scc(real, pred)
  
  mask = tf.cast(mask, dtype=loss_.dtype)
  
  loss_ *= mask

  return tf.reduce_mean(loss_)

 

  return tf.reduce_mean(loss_)     
        
from tensorflow.keras.models import Model
from keras.layers import Dense,Embedding,GRU
import tensorflow as tf

class Encoder(Model):
    def __init__(self,emb_dim,ld,voc_size_source):
        super().__init__()
        self.ld=ld
        self.embedding=Embedding(voc_size_source, emb_dim)
        self.gru_layer=GRU(ld,return_sequences=True,return_state=True)
        
    def call(self,source_tensor):
        x=self.embedding(source_tensor)
        output,hidden_state=self.gru_layer(x)
        return hidden_state
        
        
class Decoder(Model):
    def __init__(self,emb_dim,ld,voc_Target):
        super().__init__()
        self.ld=ld
        self.embedding=Embedding(voc_Target, emb_dim)
        self.gru_layer=GRU(ld,return_sequences=True,return_state=True)
        self.hs2target=Dense(voc_Target,activation='softmax')# if softmax not used  loss is hanged at pred_tsi
    # we are going to build  decoders in a way that it could handle only one time step per cal
    def call(self,target_tensor,prev_hidden_State):
        x=self.embedding(target_tensor)
        # for attention models hidden state do not affect a lot
        _,hidden_state=self.gru_layer(x,initial_state=prev_hidden_State)
        op=self.hs2target(hidden_state)
        return op,hidden_state

class Seq2Seq(Model):
    def __init__(self,emb_dim,ld,voc_Source,voc_Target):
       super().__init__()
       self.encoder=Encoder(emb_dim,ld,voc_Source)
       self.decoder=Decoder(emb_dim,ld,voc_Target)
    def predict(self,source_tensor,start_token,end_token):
      
       no_batches=int(len(source_tensor)/1)
       output=[]
       for B in range(no_batches):
               sent=[]
               source=source_tensor[B:(B+1)]
               enc_hidden_state=self.encoder(source)
               dec_prev_hidden_State=enc_hidden_state
               time_step_i_ip=[start_token]
               
               for i in range(self.seq_len-1):# end is not given as input
                   
                    time_step_i_ip=tf.expand_dims(time_step_i_ip, axis=1) # changing the the dimension from (batch size,) to (batchsize,timestep(1)) since deocder based on calculating for single time step  making in axis 1 
                    time_step_i_ip,dec_prev_hidden_State=self.decoder(time_step_i_ip,dec_prev_hidden_State)
                    
                    prediction=tf.argmax(time_step_i_ip[0],axis=0).numpy() # taking the argument ehich has the maximum probability in the hiiden dimension axis not in batch axis
                  
                    time_step_i_ip=[prediction]
                    sent.append(prediction)
                    if(prediction==end_token):
                        break
               output.append(sent)
       return output
                    
                 
    def fit(self,source_tensor,target_tensor,epochs,batch_size):
       optimiser=tf.keras.optimizers.Adam()
       assert((len(source_tensor)==len(target_tensor)) and (len(source_tensor)%batch_size==0))
       no_batches=int(len(source_tensor)/batch_size)
       self.seq_len=target_tensor.shape[1]
       for epc in range(epochs):
           total_loss=[]
           for B in range(no_batches):
               loss=0
               source=source_tensor[B*batch_size:(B+1)*batch_size]
               target=target_tensor[B*batch_size:(B+1)*batch_size]
               
               with tf.GradientTape() as tape:
                   enc_hidden_state=self.encoder(source)
                   
                   # final hidden state will be in shape batchsize,ld
                   dec_prev_hidden_State=enc_hidden_state
                   for i in range(self.seq_len-1):# end is not given as input
                       tsi=target[:,i]
                       time_step_i_ip=tf.expand_dims(tsi, axis=1)# changing the the dimension from (batch size,) to (batchsize,timestep(1)) since deocder based on calculating for single time step  making in axis 1 
                       
                       pred_tsi,dec_prev_hidden_State=self.decoder(time_step_i_ip,dec_prev_hidden_State)
                       real=target[:,i+1]# given a word it must try to predict next word
                      
                       loss+=loss_function(real, pred_tsi)
                      
                   batch_loss = (loss /self.seq_len)
                   # get all the variables
                   variables = self.encoder.trainable_variables + self.decoder.trainable_variables
                   
                   gradients = tape.gradient(loss, variables)
                   
                   optimiser.apply_gradients(zip(gradients, variables))                         
                          
                   total_loss.append(batch_loss)    
                  
                       
           print('epoch : ',epc,'  loss  :',avg(total_loss))          
        
    
dataset=pd.read_csv('chitchat.txt')


src,target=create_dataset(dataset.sample(frac = 1) , 5000,source_col='Question',target_col='Answer')

embedding_dim=256
latent_dim=1024
epochs=40
batch_Size=10

src_wordindex,src_indexword,source_tensor,voc_size_source=Tokenise(src)
tar_wordindex,tar_indexword,target_tensor,voc_size_target=Tokenise(target)

source_tensor,voc_size_source=pad_sequences(source_tensor,voc_size_source,padding='post')
target_tensor,voc_size_target=pad_sequences(target_tensor,voc_size_target,padding='post')

start=tar_wordindex['stxaxrt']
end=tar_wordindex['exnxd']

test=Seq2Seq(embedding_dim, latent_dim, voc_size_source, voc_size_target)
test.fit(source_tensor, target_tensor, epochs, batch_Size)

source_test=source_tensor[3210:3290]
test_pred=test.predict(source_test,start,end)
convert(src_indexword,tar_indexword,source_test,test_pred)

