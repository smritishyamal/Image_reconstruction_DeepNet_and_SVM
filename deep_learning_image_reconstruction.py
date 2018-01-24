#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:14:16 2018

@author: Smriti
"""

from __future__ import division
#from numpy import *
import numpy as np
#from sklearn import svm
import time
#from PIL import Image
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense

model.add(Dense(units=200, activation='relu', input_dim=3))
model.add(Dense(units=400, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=1, activation='linear'))

#rawTrainingData = np.loadtxt('abalone.txt', delimiter = ',')
rawTrainingData = np.loadtxt('abalone_challenge.txt', delimiter = ',')
lenx = len(rawTrainingData[:,1])
leny = len(rawTrainingData[1,:])
rawdata = np.zeros((lenx*leny,3))

count = 0
for i in range(lenx): #1030
    for j in range(leny): #763
        rawdata[count,0] = i
        rawdata[count,1] = j
        rawdata[count,2] = rawTrainingData[i,j]
        count = count + 1
        
rawOutputData = np.loadtxt('abalone_output_new.txt', delimiter = ',')  

rawodata = np.zeros((lenx*leny,1))
count = 0
for i in range(lenx): #1030
    for j in range(leny): #763
        #rawodata[count,int(rawOutputData[i,j])] = 1
        rawodata[count,0] = rawOutputData[i,j]
        count = count + 1
        
# Convert 0-255 to 0-1
#rawodata1 = rawodata/255        

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(rawdata, rawodata, epochs=25000, batch_size=1000) 
loss_and_metrics = model.evaluate(rawdata, rawodata, batch_size=128) 
classes = model.predict(rawdata, batch_size=128) 

classes_new = np.ceil(classes)

imageoutdata = np.zeros((lenx,leny))
count = 0
for i in range(lenx): #1030
    for j in range(leny): #763
        imageoutdata[i,j] = int(classes_new[count])
        count = count + 1

# save the output matrix here
np.savetxt('outputimagematrix_final.txt',imageoutdata)

# print matrix as image
#im = Image.fromarray(imageoutdata)
#im.show()
