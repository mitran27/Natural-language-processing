# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:19:33 2020

@author: Mitran
"""

import pandas  as pd
dataset=pd.read_csv('news_summary.csv')

import re
def process_sentence(sent):
    x=sent
    x=x.lower()
    x=re.sub("'", "", x)
    x=re.sub('\d','',x)
    x=re.sub(r'[<>@#!.]',r' ',x)
    x= re.sub(r"([?.!,¿])", r" \1 ", x)
    x=' '.join(x.split())
    x=' start_ '+x+' _end '
    return x
def create_dataset(dataset,no_Samples):
    source=[]
    target=[]
    for i in range(no_Samples):
        src=dataset.iloc[i]['text'] # ith sample 1st colums
        tar=dataset.iloc[i]['headlines']
        source.append(process_sentence(src))
        target.append(process_sentence(tar))
    return source,target


def convert(vocabulary_source,vocabulary_target,source,prediction):
    for sents,sentt in zip(source,prediction):
        sentence_source=''
        sentence_target=''
        for word_s in sents:
            if(word_s==0):
                sentence_source +=' '
            else:
               sentence_source += vocabulary_source.index_word[word_s] + ' '
           
        for word_t in sentt:
            
            if(word_t==0):
                sentence_target+=' '
            else :               
               sentence_target += vocabulary_target.index_word[word_t] + ' '
    
        print('source:  ',sentence_source,'\n\n','target  :',sentence_target,'\n\n\n')
            
            

#encoder
from keras.models import Model
from keras.layers import GRU as gru,Dense,Embedding,Bidirectional
import tensorflow as tf



from keras.losses import sparse_categorical_crossentropy as scc
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # what mask does is  if( real is 0 it is zero else 1 )
  loss_ = scc(real, pred)
  
  mask = tf.cast(mask, dtype=loss_.dtype)
  
  loss_ *= mask

  return loss_,tf.reduce_mean(loss_)

def avg(x):
    return (sum(x)/len(x))
class Encoder(Model):
    def __init__(self, ld, vocab_size,embedding_dim):
        
        super().__init__()
        self.ld=ld
        self.embedding_layer=Embedding(vocab_size, embedding_dim)
        self.gru=Bidirectional(gru(ld,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform'))
        
    def call(self,source_text):
        w2v=self.embedding_layer(source_text)       
        output,hidden_State,_=self.gru(w2v)
        return output,hidden_State
    
from tensorflow.nn import tanh,softmax
import numpy as np

class Attention(Model):
    def __init__(self,ld):
        super().__init__()
        self.W1=Dense(ld)
        self.W2=Dense(ld)
        self.V=Dense(1) # score has one value (batchsize ,1)
        
    def call(self,enocoder_hidden_states_all_time,decoder_hidden_state_prev_time):
        
        # converting deocoder hidden state to (batchsize,ld) to(batchsize,1,ld) bcz encoder has shape(batchsize,timestamp,ld)  so decoder time stamp is 1 (prev time stamp)
        decoder_hidden_state_prev_time=tf.expand_dims(decoder_hidden_state_prev_time, axis=1)
        
        # calculating the weight for decoder and encoder
        decoder_Weight=self.W1(decoder_hidden_state_prev_time)
        encoder_weight=self.W2(enocoder_hidden_states_all_time)
        
        # applying activation function and passing through dense to find scores
       
        score=self.V(tanh(decoder_Weight*encoder_weight))
        
        attention_Weight=softmax(score,axis=1)
        # first axis(0) is batch we are not going to caluclate score wrt first time stamp for all batches
        # seconde axis(1) is time stamp we are going to calculate score wrt single batch for all timestamps(hidden states)
        
        # to get the context vectore mutiply the encoders hidden states with their scores
        #  ()*(batchsize,seq_len,hidden_units)
        
        
        context_vector= attention_Weight * enocoder_hidden_states_all_time
        context_Vector=tf.reduce_sum(context_vector,axis=1)
        # add  the vectors of all time stamps so axis 1
        # axis 0 add all the batches (they are no way related with other member in batch)
        # axis 2 add all the hidden neurons
        
        return context_Vector, attention_Weight
    
    
class Decoder(Model):
    
    def __init__(self,ld,vocab_Size_target,embedding_dim):
        
        super().__init__()
        self.ld=ld
        self.embedding_layer=Embedding(vocab_Size_target, embedding_dim)
        self.gru=gru(ld,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')   
        self.hs_2_target=Dense(vocab_Size_target,activation='softmax')
        
        
    def call(self,context_Vector,prev_hidden_State,input_prevop):
       
        x=self.embedding_layer(input_prevop)# the input is passed to the embedding layer
               
        z=tf.concat([context_Vector,x],axis=-1) # the concat must be done wrt  last dimension  (hidden state dimension and the input vectors)
       
        # gru shape must be 3 dimensions else Input 0 of layer gru_13 is incompatible with the layer: expected ndim=3, found ndim=2
        hs,state=self.gru(z,initial_state=prev_hidden_State)     
       
        output=self.hs_2_target(state) # this converts the hidden states to output probabilities
        
        return output,state
    
class Seq2SeqwithATTN():
    
    def __init__(self,units,vocab_size,embedding_dim):
        super().__init__()
        [source,target]=vocab_size
        self.encoder=Encoder(units, source, embedding_dim)
        self.attention=Attention(units)
        self.decoder=Decoder(units, target, embedding_dim)
        
    def fit(self,parameters,batch_size,epochs,default_Start,optimeser='Adam'):
         optimiser=tf.keras.optimizers.Adam()
         [source_input,target_output]=parameters
         assert len(source_input)==len(target_output)# check length of input and output sequence
         assert len(source_input)%batch_size==0# check whether batch size is divisable
         no_batches=int(len(source_input)/batch_size)
        
         for i in range(epochs):             
               total_loss=[]
               
                   
               for b in range(no_batches):
                   loss=0
                   # splitting the batches for corrent b step
                   source_input_batch=source_input[b*batch_size:batch_size*(b+1),:]
                   target_output_batch=target_output[b*batch_size:batch_size*(b+1),:]
                   # starting the radients so tha tape will look for 
                   with tf.GradientTape() as tape:
                       # pass the whole sequence to encoder to get all hidden state and final hidden state
                      output_enc,decoder_hidden_State=self.encoder(source_input_batch)
                     
                      # feed start to produce inputs
                      # there must be start for all inputs of a  batches so that it will train
                      decoder_input=target_output_batch[:,0]
                      decoder_input=tf.expand_dims(decoder_input, axis=1) # encoder has a input size of batch size and time_step so change decoder to (batch_size,timestep(1))
                          
                         # Teacher forcing                
                      for t in range(len(target_output[0])-1):
                              
                          # find the context vector with hidden state and encoder outputs
                          cv,_=self.attention(output_enc,decoder_hidden_State)
                          # to pass to decoder dimension should be three  so that it could easily be concatenated with input ( shapebatchsize,1,ld) se dim 1 bcz only  for curr time step
                          # shape of x is batch,seq_len(1),vectors of words , context vector shape (batchsize ,ld) , input shape (batchsize,seqlen,embedding_dim)
                          context_Vector=tf.expand_dims(cv, axis=1)
                          # passing the necceary parameters to decoder
                          output_preictions,decoder_hidden_State=self.decoder(context_Vector,decoder_hidden_State,decoder_input)
                          #If the output is a one-hot encoded vector, then use categorical_crossentropy. 
                          # Use SparseCategoricalCrossentropy loss for word2index vector containing integers.

                          err,losses=loss_function(target_output_batch[:,t+1], output_preictions)
                          loss+=losses
                          # since we are using teacher forcing the target output is replace with acutal output which should be given as input to decoder next time step
                          decoder_input=target_output_batch[:,t+1]
                          decoder_input=tf.expand_dims(decoder_input, axis=1) # encoder has a input size of batch size and time_step so change decoder to (batch_size,timestep(1))
                   # calculatinf average batch loss       
                   batch_loss = (loss / int(target_output_batch.shape[1]))
                   # get all the variables
                   variables = self.encoder.trainable_variables + self.decoder.trainable_variables+self.attention.trainable_variables
                   
                   gradients = tape.gradient(loss, variables)
                   
                   optimiser.apply_gradients(zip(gradients, variables))                         
                          
                   total_loss.append(batch_loss)       
                  
                   
               print('epoch : ',i,'  loss  :',avg(total_loss))   
        
    def predict(self, x,default_Start,end,max_target_length):      
       
         source_input=x
         batch_size=1
         no_batches=int(len(source_input)/batch_size)
         output=[]
         for b in range(no_batches):
            sent=[]
            source_input_batch=source_input[b*batch_size:batch_size*(b+1),:]
            output_enc,decoder_hidden_State=self.encoder(source_input_batch)
            decoder_input=[start]*batch_size
            decoder_input=tf.expand_dims(decoder_input, axis=1) # encoder has a input size of batch size and time_step so change decoder to (batch_size,timestep(1))
            for t in range(max_target_length):
                 
                 cv,_=self.attention(output_enc,decoder_hidden_State)
                 # to pass to decoder dimension should be three  so that it could easily be concatenated with input ( shapebatchsize,1,ld) se dim 1 bcz only  for curr time step
                 context_Vector=tf.expand_dims(cv, axis=1)
                 
                 output_preictions,decoder_hidden_State=self.decoder(context_Vector,decoder_hidden_State,decoder_input)
                
                 prediction_id= tf.argmax(output_preictions[0]).numpy()
                 if(prediction_id==end):
                     break
                 
                 # predicted id is fed back to as input to the decoder
                 decoder_input = tf.expand_dims([prediction_id], 0)
                 sent.append(prediction_id)
            output.append(sent)
         return output
                   
                  
                   
               
            

source,target=create_dataset(dataset, 200)


from keras.preprocessing.text import Tokenizer
def pad_sequences(text,padding):
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
        
    return np.array((pad_Seq))


source_tokenizer=Tokenizer()
source_tokenizer.fit_on_texts(source)
source_tensor=source_tokenizer.texts_to_sequences(source)
source_tensor= pad_sequences(source_tensor,padding='post')

target_tokenizer=Tokenizer()
target_tokenizer.fit_on_texts(target)
start=target_tokenizer.word_index['start']
end=target_tokenizer.word_index['end']
target_tensor=target_tokenizer.texts_to_sequences(target)

target_tensor= pad_sequences(target_tensor,padding='post')





max_source_length=max([len(i.split()) for i in source])
max_target_length=max([len(i.split()) for i in target])

embedding_dim=512
latent_dim=1024
epochs=100
batch_Size=2

vocab_size_source=len(source_tokenizer.word_index)+1 # one for the padding of zeros
vocab_Size_target=len(target_tokenizer.word_index)+1


test=Seq2SeqwithATTN(latent_dim,[vocab_size_source,vocab_Size_target],embedding_dim)
test.fit([source_tensor,target_tensor],batch_Size,100,start)

source_test=source_tensor[350:360,:]
test_op=test.predict(source_test,start,end,max_target_length)
convert(source_tokenizer,target_tokenizer,source,test_op)
        
    