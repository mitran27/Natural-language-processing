# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:07:27 2020
speech recognition
@author: Mitran
"""


import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")
sr=16000
train_audio_path = './train/audio/'
samples, sample_rate = librosa.load(train_audio_path+'yes/0a7c2a8d_nohash_0.wav', sr)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave ')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot()
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

ipd.Audio(samples, rate=sample_rate)

newsr=8000
samples=librosa.resample(samples,sr,newsr)# samples are divided based on sr and new sr they are spliited as sr/newsr
print(sample_rate,len(samples))            #samples(sr/newsr)
labels=os.listdir(train_audio_path)
#print(labels)
no_of_recor=[]
for i in labels:
   wave=[aud for aud in os.listdir(train_audio_path+'/'+i) if aud.endswith('.wav')]
   no_of_recor.append(len(wave))


plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recor)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.title('No. of recordings for each command')
plt.show()
"""
duration=[]
for i in labels:
   wave=[aud for aud in os.listdir(train_audio_path+'/'+i) if aud.endswith('.wav')]
   for wav in wave:
       samplerate,sample=wavfile.read(train_audio_path+'/'+i+'/'+wav)
       duration.append(float(len(sample)/samplerate))
plt.hist(np.array(duration))
"""
duration=[]
labelsall=[]
wavesall=[]
for count,i in enumerate(labels):
   print(i)
   wave=[aud for aud in os.listdir(train_audio_path+'/'+i) if aud.endswith('.wav')]
   for wav in wave:
       #samplerate,sample=wavfile.read(train_audio_path+'/'+i+'/'+wav)
       sample, samplerate = librosa.load(train_audio_path+'/'+i+'/'+wav, 16000)
   
       #
     
       if(len(sample)==16000):
          sample=librosa.resample(sample,samplerate,8000)
          duration.append(float(len(sample)/samplerate))
          wavesall.append(sample)
          labelsall.append(i)
plt.hist(np.array(duration))


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(labelsall)
classes= list(le.classes_)


from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))

all_wave = np.array(wavesall).reshape(-1,8000,1)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.8,random_state=777,shuffle=True)

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)



model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')




history=model.fit(xtrain, ytrain ,epochs=20, callbacks=[es,mc], batch_size=32, validation_data=(xtest,ytest))
def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


import sounddevice as sd
import soundfile as sf

samplerate = 16000  
duration = 1 # seconds
filename = 'yes.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)



#reading the voice commands
samples, sample_rate = librosa.load('./' + 'yes.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples,rate=8000)  

print(predict(samples))