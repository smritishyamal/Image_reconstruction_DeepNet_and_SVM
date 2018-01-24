#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:14:16 2018

@author: Smriti
"""

from __future__ import division
#from numpy import *
import numpy as np
from sklearn import svm
import time
from PIL import Image

# SVM training
# code snipet
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
        
#X = [[0], [1], [2], [3]]

rawOutputData = np.loadtxt('abalone_output_new.txt', delimiter = ',')  

rawodata = []#np.zeros((lenx*leny,1))
#count = 0
for i in range(lenx): #1030
    for j in range(leny): #763
        rawodata.append(int(rawOutputData[i,j]))

#Y = [0, 1, 2, 3]
#X = [[0,1,2], [1,4,3], [2,0,1], [3,3,1]]
# We have used linear support vector machine for classification problem
svmtrain = svm.SVR(C=1, kernel = 'rbf', shrinking = False, max_iter=100000)
        

#clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
#    max_iter=1, probability=False, random_state=None, shrinking=True,
#    verbose=True)

#clf.fit(rawdata,rawodata) 
svmtrain.fit(rawdata,rawodata) 
classes = np.zeros(len(rawodata))

for i in range(103):
    # 10 rows at a time
    classes[7630*i:7630*(i+1)] = svmtrain.predict(rawdata[7630*i:7630*(i+1),:])
    
#classes = svmtrain.predict(rawdata[0:,:])

classes_new = np.ceil(classes)

imageoutdata = np.zeros((lenx,leny))
count = 0
for i in range(lenx): #1030
    for j in range(leny): #763
        imageoutdata[i,j] = int(classes_new[count])
        count = count + 1
        
# print matrix as image
#im = Image.fromarray(imageoutdata)
#im.show()

np.savetxt('outputimagematrix_svm_svr_100000.txt',imageoutdata)
