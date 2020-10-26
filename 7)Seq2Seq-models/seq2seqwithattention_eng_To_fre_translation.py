data_path='fra.txt'

import pandas  as pd
dataset=pd.read_table(data_path,names=['source','target','comments'])
dataset=dataset.drop('comments',axis=1)
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
        x=dataset.iloc[i,:]
        src=x['source']
        tar=x['target']
        source.append(process_sentence(src))
        target.append(process_sentence(tar))
    return source,target


def convert(vocabulary_source,vocabulary_target,source,prediction):
    for sents,sentt in zip(source,prediction):
        sentence_source=''
        sentence_target=''
        for word_s,word_t in zip(sents,sentt):
            if(word_s==0):
                sentence_source +=' '
            else:
               sentence_source += vocabulary_source.index_word[word_s] + ' '
            if(word_t==0):
                sentence_target+=' '
            else :               
               sentence_target += vocabulary_target.index_word[word_t] + ' '
            
        print(sentence_source,'\n',sentence_target,'\n\n\n')
            
            

#encoder
from keras.models import Model
from keras.layers import GRU as gru,Dense,Embedding
import tensorflow as tf
from keras.losses import sparse_categorical_crossentropy as scc



def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = scc(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

class Encoder(Model):
    def __init__(self, ld, vocab_size,embedding_dim):
        
        super().__init__()
        self.ld=ld
        self.embedding_layer=Embedding(vocab_size, embedding_dim)
        self.gru=gru(ld,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        
    def call(self,source_text):
       
        w2v=self.embedding_layer(source_text)
        output,hidden_State=self.gru(w2v)
        return output,hidden_State
    
from tensorflow.nn import tanh,softmax

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
        score=self.V(tanh(decoder_Weight+encoder_weight))
        
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
        
        # shape of x is batch,seq_len,vectors of words
        # context vector shape (batchsize ,ld)
        # input shape (batchsize,seqlen,embedding_dim)
       
        context_Vector=tf.expand_dims(context_Vector, axis=1)
       
        z=tf.concat([context_Vector,x],axis=-1) # the concat must be done wrt  last dimension  (hidden state dimension and the input vectors)
        
        hidden_states,state=self.gru(z)
        
        # output is three dimension
        # batch size,1(time step),ld+embed_dim
        # to convert to2 dim to pass throu forward layer
        batch_size=hidden_states.shape[0]
        hidden_states_forcurr_timestep=tf.reshape(hidden_states,(batch_size,-1))
        
        
        output=self.hs_2_target(hidden_states_forcurr_timestep)
        
        return output,state
    
class Seq2SeqwithATTN(Model):
    
    def __init__(self,units,vocab_size,embedding_dim):
        super().__init__()
        [source,target]=vocab_size
        self.encoder=Encoder(units, source, embedding_dim)
        self.attention=Attention(units)
        self.decoder=Decoder(units, target, embedding_dim)
        """If the output is a one-hot encoded vector, then use categorical_crossentropy. 
        Use SparseCategoricalCrossentropy loss for word2index vector containing integers.
"""
    def fit(self,parameters,batch_size,epochs,default_Start,optimeser='Adam'):
         optimiser=tf.keras.optimizers.Adam()
         [source_input,target_output]=parameters
         assert len(source_input)==len(target_output)
         assert len(source_input)%batch_size==0
         no_batches=int(len(source_input)/batch_size)
        
         for i in range(epochs):             
               total_loss=0
               
                   
               for b in range(no_batches):
                   loss=0
                  
                   source_input_batch=source_input[b*batch_size:batch_size*(b+1),:]
                   target_output_batch=target_output[b*batch_size:batch_size*(b+1),:]
                   with tf.GradientTape() as tape:
                      output_enc,decoder_hidden_State=self.encoder(source_input_batch)
                     
                      # feed start to produce inputs
                      # there must be start for all inputs of a  batches
                      decoder_input=[start]*batch_size
                      decoder_input=tf.expand_dims(decoder_input, axis=1) # encoder has a input size of batch size and time_step so change decoder to (batch_size,timestep(1))
                          
                         # Teacher forcing                
                      for t in range(len(target_output[0])-1):
                              
                          # find the context vector with hidden state and encoder outputs
                          cv,_=self.attention(output_enc,decoder_hidden_State)
                          output_preictions,decoder_hidden_State=self.decoder(cv,decoder_hidden_State,decoder_input)
                          
                          loss+=loss_function(target_output_batch[:,t+1], output_preictions)
                          
                          decoder_input=target_output_batch[:,t+1]
                          decoder_input=tf.expand_dims(decoder_input, axis=1) # encoder has a input size of batch size and time_step so change decoder to (batch_size,timestep(1))
                          
                   batch_loss = (loss / int(target_output_batch.shape[1]))
                   variables = self.encoder.trainable_variables + self.decoder.trainable_variables+self.attention.trainable_variables
                          
                   gradients = tape.gradient(loss, variables)
                   optimiser.apply_gradients(zip(gradients, variables))
                  
                          
                          
                   total_loss+=batch_loss        
                  
                   
               print('epoch : ',i,'  loss  :',(total_loss))   
        
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
                 
                 output_preictions,decoder_hidden_State=self.decoder(cv,decoder_hidden_State,decoder_input)
                
                 prediction_id= tf.argmax(output_preictions[0]).numpy()
                 if(prediction_id==end):
                     break
                 
                 # predicted id is fed back to as input to the decoder
                 decoder_input = tf.expand_dims([prediction_id], 0)
                 sent.append(prediction_id)
            output.append(sent)
         return output
                   
                  
                   
               
            

source,target=create_dataset(dataset, 1000)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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

embedding_dim=256
latent_dim=1024
epochs=300
batch_Size=50

vocab_size_source=len(source_tokenizer.word_index)+1 # one for the padding of zeros
vocab_Size_target=len(target_tokenizer.word_index)+1

"""tes=Encoder(latent_dim, vocab_size_source, embedding_dim)
op,hs=tes(source_tensor)
test=Attention(latent_dim)
cv,atwt=test(op,hs)
"""
source=source_tensor[950:1000,:]
test=Seq2SeqwithATTN(latent_dim,[vocab_size_source,vocab_Size_target],embedding_dim)
test.fit([source_tensor,target_tensor],batch_Size,epochs,start)
test_op=test.predict(source,start,end,max_target_length)

convert(source_tokenizer,target_tokenizer,source,test_op)










    
    

