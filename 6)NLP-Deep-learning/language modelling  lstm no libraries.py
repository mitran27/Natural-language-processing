import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    e_x = np.exp(x - np.max(x)) # max(x) subtracted for numerical stability
    return e_x / np.sum(e_x)
class LSTM:
    def __init__(self, char_to_idx, idx_to_char, vocab_size, n_h=100, seq_len=25,epochs=10, lr=0.01, ):
        self.char_id=char_to_idx
        self.id_char=idx_to_char
        self.word_count=vocab_size
        self.n_h=n_h
        self.seq_len=seq_len
        
        
        self.param={}
        x=self.word_count+self.n_h# mostly use variable
        n_h=self.n_h
        xav=(1.0/np.sqrt(x))# Xavier initialisation
        
        self.param['wf']=np.random.randn(n_h,x)*xav
        self.param['bf']=np.ones((n_h,1))
        
        self.param['wi']=np.random.randn(n_h,x)*xav
        self.param['bi']=np.ones((n_h,1))
        
        self.param['wc']=np.random.randn(n_h,x)*xav
        self.param['bc']=np.ones((n_h,1))
        
        self.param['wo']=np.random.randn(n_h,x)*xav
        self.param['bo']=np.ones((n_h,1))
        
        self.param['wop']=np.random.randn(self.word_count, self.n_h)* (1.0/np.sqrt(self.word_count))
        self.param['bop']=np.ones((self.word_count,1))
        
        
        self.grad = {}
        self.adam_params = {}
        for i in self.param:
            self.grad['d'+i]=np.zeros((self.param[i].shape))
            self.adam_params['m'+i]=np.zeros((self.param[i].shape))
            self.adam_params['adam'+i]=np.zeros((self.param[i].shape))
        self.smooth_loss = -np.log(1.0 / self.word_count) * self.seq_len
        return
    
    def forward_propagation(self, x, h_prev, c_prev): 
        #print(h_prev.shape,x.shape)
        z=np.row_stack((h_prev, x))
       
        #print(self.param['wf'].shape,z.shape)
        
        f=sigmoid(np.dot(self.param['wf'],z)+self.param['bf'])
        
        i=sigmoid(np.dot(self.param['wi'],z)+self.param['bi'])
        c_bar=np.tanh(np.dot(self.param['wc'],z)+self.param['bc'])
        c = f * c_prev + i * c_bar
        
        
        o = sigmoid(np.dot(self.param["wo"], z) + self.param["bo"])
        h = o * np.tanh(c)
        
        
        
        
        v = np.dot(self.param["wop"], h) + self.param["bop"]       
        y_hat = softmax(v)
        
        return [y_hat, v, h, o, c, c_bar, i, f, z]
    
    def clip_grads(self):
      for key in self.grad:
          np.clip(self.grad[key], -5, 5, out=self.grad[key])
    def reset_grads(self):
      for key in self.grad:
          self.grad[key].fill(0)
      return
    def update_param(self,batch_num,lr):
        beta1=0.9
        beta2=0.999
        for key in self.param:
            self.adam_params['m'+key]=beta1*self.adam_params['m'+key]+(1-beta1)*self.grad['d'+key]
            self.adam_params['adam'+key]=beta2*self.adam_params['adam'+key]+(1-beta2) * self.grad["d"+key]**2
            
            
            m_correlated = self.adam_params["m" + key] / (1 - beta1**batch_num)
            v_correlated = self.adam_params["adam" + key] / (1 - beta2**batch_num) 
            self.param[key] -=lr * m_correlated / (np.sqrt(v_correlated) + 1e-8) 
            
    def back_propagation(self,y,y_hat,dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
        dv=y_hat.copy()
        dv[y]-=1  # calculating error mse error of current cell not the iterated cell see the image of forward propagation
        
        
        """1. As weights are shared by all time steps, the weight gradients are accumulated.
        
           2. We are adding dh_next to dh, because as Figure 1 shows,
           h is branched in forward propagation in the softmax output layer and the next LSTM cell,
           where it is concatenated with x. Therefore, 
           there are two gradients flowing back.
           This applies to dc as well.
           
           3. There are four gradients flowing towards the input layer from the gates,
           therefore dz is the summation of those gradients."""       
        
        
        
        
        
        #from the output
        self.grad['dwop']+=np.dot(dv,h.T) # 1 see note
        self.grad['dbop']+=dv # 2
        # hidden state
        dh = np.dot(self.param["wop"].T, dv) # 3
        dh+=dh_next # accumulating the traelling parameters
        
        ##########-----output gate-----###########
        do = dh * np.tanh(c) # 4
        d_ao=do*o*(1-o) # 5
        
        self.grad["dwo"] += np.dot(d_ao, z.T) #6
        self.grad["dbo"] += d_ao# 7
        
        # cell state
        
        dc = dh * o * (1-np.tanh(c)**2) #8
        dc += dc_next #9
        
        dc_bar=dc* i#10
        dac=dc_bar*(1-c_bar**2)
        
        self.grad["dwc"] += np.dot(dac, z.T) #11
        self.grad["dbc"] += dac# 12
       
        
        ##########-----input gate-----###########
        
        di = dc * c_bar#13
        da_i = di * i*(1-i) #14
        self.grad["dwi"] += np.dot(da_i, z.T)#15
        self.grad["dbi"] += da_i#16
        
        dc_prev = f * dc#22
        
         ##########-----forget gate-----###########
        
        df = dc * c_prev  #17
        da_f = df * f*(1-f)#18
        self.grad["dwf"] += np.dot(da_f, z.T)#19
        self.grad["dbf"] += da_f#20
        
        
        
        dz = (np.dot(self.param["wf"].T, da_f)+ np.dot(self.param["wi"].T, da_i)+ np.dot(self.param["wc"].T, dac)+ np.dot(self.param["wo"].T, d_ao))   #As weights are shared by all time steps, the weight gradients are accumulated.
        dh_prev = dz[:self.n_h, :]  #z=np.hstack((h_prev,x))  #21
        
        
        return dh_prev, dc_prev
    
    
    def fit(self, x_batch, y_batch, h_prev, c_prev):
        x, z = {}, {}
        f, i, c_bar, c, o = {}, {}, {}, {}, {}
        y_hat, v, h = {}, {}, {}
        # Values at t= - 1
        h[-1] = h_prev
        c[-1] = c_prev
        
        loss =0
        for t in range (self.seq_len):
            x[t]=np.zeros((self.word_count,1))
            x[t][x_batch[t]]=1
            nxt= self.forward_propagation(x[t], h[t-1], c[t-1])
            #print('asd',len(nxt))
            y_hat[t]=nxt[0]
            v[t]=nxt[1] 
            h[t]=nxt[2] 
            o[t]=nxt[3] 
            c[t]=nxt[4] 
            c_bar[t]=nxt[5] 
            i[t]=nxt[6] 
            f[t]=nxt[7] 
            z[t] = nxt[8]
            #print('goi')  
           
            loss+=-np.log(y_hat[t][y_batch[t],0])
           
            
        self.reset_grads()
        
        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])
            
        for t in reversed(range(self.seq_len)):
               dh_next, dc_next = self.back_propagation(y_batch[t], y_hat[t], dh_next, dc_next, c[t-1], z[t], f[t], i[t],c_bar[t], c[t], o[t], h[t]) 
        return loss, h[self.seq_len-1], c[self.seq_len-1]
    
    def train(self, X, verbose=True):
      J = []  # to store losses

      num_batches = len(X) // self.seq_len
      X_trimmed = X[: num_batches * self.seq_len]  # trim input to have full sequences
     
      epch=20
      #print('go')
      for epoch in range(epch):
          h_prev = np.zeros((self.n_h, 1))
          c_prev = np.zeros((self.n_h, 1))
          #print('go')
          for j in range(0, len(X_trimmed) - self.seq_len, self.seq_len):
              # prepare batches
              x_batch = [self.char_id[ch] for ch in X_trimmed[j: j + self.seq_len]]
              y_batch = [self.char_id[ch] for ch in X_trimmed[j + 1: j + self.seq_len + 1]]
              
              loss, h_prev, c_prev = self.fit(x_batch, y_batch, h_prev, c_prev)
              #print('go')
              # smooth out loss and store in list
              self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
              J.append(self.smooth_loss)
  
              # check gradients
              if epoch == 0 and j == 0:
                  self.gradient_check(x_batch, y_batch, h_prev, c_prev, num_checks=10, delta=1e-7)
              
              self.clip_grads()
  
              batch_num = epoch * epch + j / self.seq_len + 1
              self.update_param(batch_num,0.001)
              
             # print out loss and sample string
              if verbose:
                  if j % 400000 == 0:
                      print('Epoch:', epoch, '\tBatch:', j, "-", j + self.seq_len,'\tLoss:', round(self.smooth_loss, 2))
                      s = self.sample(h_prev, c_prev, sample_size=250)
                      print(s, "\n")
                     
      return J, self.param
  
    def sample(self, h_prev, c_prev, sample_size):
       x = np.zeros((self.word_count, 1))
       h = h_prev
       c = c_prev
       sample_string = "" 
    
       for t in range(sample_size):
           y_hat, _, h, _, c, _, _, _, _ = self.forward_propagation(x, h, c)        
         
           # get a random index within the probability distribution of y_hat(ravel())
           idx = np.random.choice(range(self.word_count), p=y_hat.ravel())
           x = np.zeros((self.word_count, 1))
           x[idx] = 1
           
           #find the char with the sampled index and concat to the output string
           char = self.id_char[idx]
           sample_string += char
       return sample_string
   
    def gradient_check(self, x, y, h_prev, c_prev, num_checks=10, delta=1e-6):
        """
        Checks the magnitude of gradients against expected approximate values
        """
        #print("**********************************")
       # print("Gradient check...\n")

        _, _, _ = self.fit(x, y, h_prev, c_prev)
        grads_numerical = self.grad

        for key in self.param:
           # print("---------", key, "---------")
            test = True

            dims = self.param[key].shape
            grad_numerical = 0
            grad_analytical = 0

            for _ in range(num_checks):  # sample 10 neurons

                idx = int(random.uniform(0, self.param[key].size))
                old_val = self.param[key].flat[idx]

                self.param[key].flat[idx] = old_val + delta
                J_plus, _, _ = self.fit(x, y, h_prev, c_prev)

                self.param[key].flat[idx] = old_val - delta
                J_minus, _, _ = self.fit(x, y, h_prev, c_prev)

                self.param[key].flat[idx] = old_val

                grad_numerical += (J_plus - J_minus) / (2 * delta)
                grad_analytical += grads_numerical["d" + key].flat[idx]

            grad_numerical /= num_checks
            grad_analytical /= num_checks

            rel_error = abs(grad_analytical - grad_numerical) / abs(grad_analytical + grad_numerical)

            if rel_error > 1e-2:
                if not (grad_analytical < 1e-6 and grad_numerical < 1e-6):
                    test = False
                    assert (test)

          
        return
    
data = open('HP1.txt').read().lower()

# converting data to set for givng it id

vocabulary_data=set( data)
print(len(data))
vocab_size=len(vocabulary_data)
print(vocab_size,len(data))
char_to_idx = {w: i for i,w in enumerate(vocabulary_data)}
idx_to_char = {i: w for i,w in enumerate(vocabulary_data)}
model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs = 5, lr = 0.01)

J, params = model.train(data)

