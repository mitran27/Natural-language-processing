# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:12:23 2020

@author: Mitran
"""


# determining positive or negative reviewen

# “many to one” RNN      Since this is a classification problem,

#but it only uses the final hidden state to produce the one output yy:


train_data = {
  'good': True,
  'bad': False,
  'happy': True,
  'sad': False,
  'not good': False,
  'not bad': True,
  'not happy': False,
  'not sad': True,
  'very good': True,
  'very bad': False,
  'very happy': True,
  'very sad': False,
  'i am happy': True,
  'this is good': True,
  'i am bad': False,
  'this is bad': False,
  'i am sad': False,
  'this is sad': False,
  'i am not happy': False,
  'this is not good': False,
  'i am not bad': True,
  'this is not sad': True,
  'i am very happy': True,
  'this is very good': True,
  'i am very bad': False,
  'this is very sad': False,
  'this is very happy': True,
  'i am good not bad': True,
  'this is good not bad': True,
  'i am bad not good': False,
  'i am good and happy': True,
  'this is not good and not happy': False,
  'i am not at all good': False,
  'i am not at all bad': True,
  'i am not at all happy': False,
  'this is not at all sad': True,
  'this is not at all happy': False,
  'i am good right now': True,
  'i am bad right now': False,
  'this is bad right now': False,
  'i am sad right now': False,
  'i was good earlier': True,
  'i was happy earlier': True,
  'i was bad earlier': False,
  'i was sad earlier': False,
  'i am very bad right now': False,
  'this is very good right now': True,
  'this is very sad right now': False,
  'this was bad earlier': False,
  'this was very good earlier': True,
  'this was very bad earlier': False,
  'this was very happy earlier': True,
  'this was very sad earlier': False,
  'i was good and not bad earlier': True,
  'i was not good and not happy earlier': False,
  'i am not at all bad or sad right now': True,
  'i am not at all good or happy right now': False,
  'this was not happy and not good earlier': False,
}

test_data = {
  'this is happy': True,
  'i am good': True,
  'this is not happy': False,
  'i am not good': False,
  'this is not bad': True,
  'i am not sad': True,
  'i am very good': True,
  'this is very bad': False,
  'i am very sad': False,
  'this is bad not good': False,
  'this is good and happy': True,
  'i am not good and not happy': False,
  'i am not at all sad': True,
  'this is not at all good': False,
  'this is not at all bad': True,
  'this is good right now': True,
  'this is sad right now': False,
  'this is very bad right now': False,
  'this was good earlier': True,
  'i was not happy and not good earlier': False,
}
import numpy as np

def softmax(x):
  # Applies the Softmax Function to the input array.
  return np.exp(x) / sum(np.exp(x))


        

vocab= list(set(w for text in train_data.keys() for w in text.split(' ')))

vocab_size=len(vocab)

word_id={i:w for i,w in enumerate(vocab)}
id_word={w:i for i,w in enumerate(vocab)}
def onhotencode(array):
    inputs = []
    for w in array.split(' '):
       v = np.zeros((vocab_size, 1))
      
       v[id_word[w]] = 1
       inputs.append(v)
    return inputs

#We can now represent any given word with its corresponding integer index!


class Rnnlayer:
    
    def __init__(self,input_size,output_size,hidden_layer_size):
        #one layer of rnnn in multiple bcsz it contins layers
        self.hls=hidden_layer_size
       # there are three weight and 2 bias for one layer of rnn which contain many recurent layers which use same weights hence callreccurent neural network
        self.wt_ip=np.random.randn(hidden_layer_size,input_size)/1000
        self.wt_prevstate=np.random.randn(hidden_layer_size,hidden_layer_size)/1000
        self.wt_op=np.random.randn(output_size,hidden_layer_size)/1000
        
        self.bias_newstate=np.zeros((hidden_layer_size,1))
        self.bias_op=np.zeros((output_size,1))
    def rnnactiv(self,x):
           return np.tanh(x);

    def forward_propagate(self,input_data):
       h = np.zeros((self.hls, 1))
       self.input=input_data
       self.stateslst={0:h}
       for i,x in enumerate(input_data):
           # initially weight for prev state are zero
          
           temp_state=self.wt_ip @ x+self.wt_prevstate @ h +self.bias_newstate    # prev state storing variable
           
           h=self.rnnactiv(temp_state)
           self.stateslst[i+1]=h
       # reccurent layers will have inbuilt activation functions
       o_p=self.wt_op@h+self.bias_op
       
       return o_p
       
       
    def back_propagate(self,err,lr):
         n = len(self.input)  # Calculate dL/dWhy and dL/dby.
         d_Why = err @ self.stateslst[n].T
         
         
         
         d_Whh = np.zeros(self.wt_prevstate.shape)
         d_Wxh = np.zeros(self.wt_ip.shape)
         d_bh = np.zeros(self.bias_newstate.shape)

    # Calculate dL/dh for the last h.
         d_h = self.wt_op.T @ err
         d_by = err
         for t in reversed(range(n)):
            temp = ((1 - self.stateslst[t + 1] ** 2) *d_h)
            
         # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.stateslst[t].T

      # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.input[t].T

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.wt_prevstate @ temp# accumulating the travelling parameters
            
            
            
            

    # Clip to prevent exploding gradients.
         for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
               np.clip(d, -1, 1, out=d)
               
         self.wt_prevstate -= lr * d_Whh
         self.wt_ip -= lr * d_Wxh
         self.wt_op -= lr * d_Why
         self.bias_newstate -= lr * d_bh
         self.bias_op -= lr * d_by



         
         
        
        
        
    
    

import random
def Mse(yHat, z):
  
   if (z==0):
       y=[[1],[0]]
   else:
       y=[[0],[1]]
   cx=np.array(y)   
   return  yHat - cx
   """
   x=yHat
   x[y]-=1
   return x
   """
rnn=Rnnlayer(vocab_size, 2,64)
def processData(data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = onhotencode(x)
    target = int(y)

    # Forward
    out = rnn.forward_propagate(inputs)
    probs = softmax(out)
    
    # Calculate loss / accuracy


    if backprop:
      # Build dL/dy
      
      """
      d_L_d_y = probs
      d_L_d_y[target] -= 1
      """
      d_L_d_y=Mse(probs,target)
      # Backward
      rnn.back_propagate(d_L_d_y,0.001)
    else:
       i=probs
           
       if(i[0]>i[1]):
              print(x,'bad')
       elif(i[0]==i[1]):
              print(x,'doubt')
       else:
           print(x,'good')
                
         

  return loss / len(data), num_correct / len(data)

for epoch in range(10000):
  train_loss, train_acc = processData(train_data)
  print(10000-epoch)


   
test_loss, test_acc = processData(test_data, backprop=False)



   
