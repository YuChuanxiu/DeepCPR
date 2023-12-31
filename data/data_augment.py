# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:00:56 2023

@author: ZNDX001
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binom, norm, beta, expon
import random
from pylab import vlines
from numpy import float32
import math
import scipy.special as sse
from scipy import signal
import matplotlib


# matplotlib.use('Qt5Agg')

def add_baseline(chrom, scans):
    baseline =  np.zeros(chrom.shape)
    for k in range(chrom.shape[1]):
        typ = np.random.randint(0,3)
        if typ==0:    
            xx = np.arange(0, scans, 1)
            Threshold_baseline = np.random.uniform(0, 10)
            baseline[:,k] = xx*Threshold_baseline  
        elif typ==1:  
            xx = np.arange(scans, 0, -1)
            Threshold_baseline = np.random.uniform(0, 10)
            baseline[:,k] = xx*Threshold_baseline            
        else:
            xx = np.ones((scans))
            Threshold_baseline = np.random.uniform(0, 10)
            baseline[:,k] = xx*Threshold_baseline 
            
    return chrom + baseline

def gauss_noise(signal, SNR):
    noise = np.random.randn(signal.shape[0],signal.shape[1]) 	
    noise = noise-np.mean(noise) 								
    signal_power = np.linalg.norm(signal - signal.mean())**2 / signal.size	
    noise_variance = signal_power/np.power(10,(SNR/10))         
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    
    signal_noise = noise + signal

    return signal_noise

def process(X):
    for i in range(X.shape[0]):   
        if np.max(X[i,:])!=0:
            X[i,:] = 10000*X[i,:]/np.max(X[i,:])
        else:
            print(i,'k')
    return X

def gauss5(loc0,spectrum,point,xa,R_range,S_range): 
    A = np.random.uniform(5e3, 8e4, (5))      
    R = np.random.uniform(R_range[0], R_range[1], (4))
    Scale = np.random.uniform(S_range[0], S_range[1], (5))

    CC = np.zeros((5,point), dtype=float32)
    for i in range(5):
        rv = norm(loc = loc0, scale = Scale[i])
        CC[i,:] = rv.pdf(xa)
             
    lw0 = loc0 - np.min(np.where((CC[0,:] > (max(CC[0,:] * 1e-3)))))
    rw0 = np.max(np.where((CC[0,:] > (max(CC[0,:] * 1e-3))))) - loc0
    lw1 = loc0 - np.min(np.where((CC[1,:] > (max(CC[1,:] * 1e-3)))))
    rw1 = np.max(np.where((CC[1,:] > (max(CC[1,:] * 1e-3))))) - loc0
    lw2 = loc0 - np.min(np.where((CC[2,:] > (max(CC[2,:] * 1e-3)))))
    rw2 = np.max(np.where((CC[2,:] > (max(CC[2,:] * 1e-3))))) - loc0
    lw3 = loc0 - np.min(np.where((CC[3,:] > (max(CC[3,:] * 1e-3)))))
    rw3 = np.max(np.where((CC[3,:] > (max(CC[3,:] * 1e-3))))) - loc0
    lw4 = loc0 - np.min(np.where((CC[4,:] > (max(CC[4,:] * 1e-3)))))
    rw4 = np.max(np.where((CC[4,:] > (max(CC[4,:] * 1e-3))))) - loc0

    loc1 = int(R[0]*(lw0+lw1)+loc0)
    loc2 = int(R[1]*(lw1+lw2)+loc1)
    loc3 = int(R[2]*(lw2+lw3)+loc2)
    loc4 = int(R[3]*(lw3+lw4)+loc3)
    loc = [loc0, loc1, loc2, loc3, loc4]
    
    C = np.zeros((5,point), dtype=float32)
    for i in range(5):
        rv = norm(loc = loc[i], scale = Scale[i])
        C[i,:] = rv.pdf(xa) * A[i] 

    index = random.sample(range(0,spectrum.shape[0]),5)
    S = (spectrum[index,:])
    chrom = np.dot(C.T,S)

    chrom = gauss_noise(chrom, np.random.randint(8,40))
    chrom = add_baseline(chrom, point)
    chrom = signal.detrend(chrom)
    
    chrom[chrom<0] = 0
    chrom = chrom/np.max(chrom)
    
    are = np.zeros((5,point), dtype=float32)
    ta = np.zeros((1,point), dtype=float32)
    are[0, math.floor(loc0-lw0):math.ceil(loc0+rw0)] = 1
    are[1, math.floor(loc1-lw1):math.ceil(loc1+rw1)] = 1
    are[2, math.floor(loc2-lw2):math.ceil(loc2+rw2)] = 1
    are[3, math.floor(loc3-lw3):math.ceil(loc3+rw3)] = 1
    are[4, math.floor(loc4-lw4):math.ceil(loc4+rw4)] = 1
    for i in range(point):
        ta[0,i] = are[0,i] + are[1,i] + are[2,i] + are[3,i] + are[4,i]
        if ta[0,i] >= 1:
            ta[0,i] = 1
    a = np.where(ta[0]==1)
    site_st = math.floor(np.min(a)-3)
    site_ed = math.ceil(np.max(a)+3)

    dis_st = math.floor((point-(site_ed-site_st))/2)
                              
    label = np.zeros((n_c,point), dtype=float32)
    label[0:5, int(dis_st):int(dis_st+site_ed-site_st)] = C[:, site_st:site_ed]
    label = label/np.max(label)

    GMdata = np.zeros((point,800), dtype=float32)    
    GMdata[int(dis_st):int(dis_st+site_ed-site_st), :] = chrom[site_st:site_ed, :]
    
    return GMdata,label,S

def gauss4(loc0,spectrum,point,xa,R_range,S_range): 
    A = np.random.uniform(5e3, 8e4, (4))      
    R = np.random.uniform(R_range[0], R_range[1], (3))
    Scale = np.random.uniform(S_range[0], S_range[1], (4))
    CC = np.zeros((4,point), dtype=float32)
    for i in range(4):
        rv = norm(loc = loc0, scale = Scale[i])
        CC[i,:] = rv.pdf(xa)

    lw0 = loc0 - np.min(np.where((CC[0,:] > (max(CC[0,:] * 1e-3)))))
    rw0 = np.max(np.where((CC[0,:] > (max(CC[0,:] * 1e-3))))) - loc0
    lw1 = loc0 - np.min(np.where((CC[1,:] > (max(CC[1,:] * 1e-3)))))
    rw1 = np.max(np.where((CC[1,:] > (max(CC[1,:] * 1e-3))))) - loc0
    lw2 = loc0 - np.min(np.where((CC[2,:] > (max(CC[2,:] * 1e-3)))))
    rw2 = np.max(np.where((CC[2,:] > (max(CC[2,:] * 1e-3))))) - loc0
    lw3 = loc0 - np.min(np.where((CC[3,:] > (max(CC[3,:] * 1e-3)))))
    rw3 = np.max(np.where((CC[3,:] > (max(CC[3,:] * 1e-3))))) - loc0

    loc1 = int(R[0]*(lw0+lw1)+loc0)
    loc2 = int(R[1]*(lw1+lw2)+loc1)
    loc3 = int(R[2]*(lw2+lw3)+loc2)
    loc = [loc0, loc1, loc2, loc3]
    
    C = np.zeros((4,point), dtype=float32)
    for i in range(4):
        rv = norm(loc = loc[i], scale = Scale[i])
        C[i,:] = rv.pdf(xa) * A[i] 

    index = random.sample(range(0,spectrum.shape[0]),4)
    S = (spectrum[index,:])
    chrom = np.dot(C.T, S)
    
    chrom = gauss_noise(chrom, np.random.randint(8,30))
    chrom = add_baseline(chrom, point)
    chrom = signal.detrend(chrom)
    chrom[chrom<0] = 0
    chrom = chrom/np.max(chrom)
    
    are = np.zeros((4,point), dtype=float32)
    ta = np.zeros((1,point), dtype=float32)
    are[0, math.floor(loc0-lw0):math.ceil(loc0+rw0)] = 1
    are[1, math.floor(loc1-lw1):math.ceil(loc1+rw1)] = 1
    are[2, math.floor(loc2-lw2):math.ceil(loc2+rw2)] = 1
    are[3, math.floor(loc3-lw3):math.ceil(loc3+rw3)] = 1
    for i in range(point):
        ta[0,i] = are[0,i] + are[1,i] + are[2,i] + are[3,i]
        if ta[0,i] >= 1:
            ta[0,i] = 1
    a = np.where(ta[0]==1)
    site_st = math.floor(np.min(a)-3)
    site_ed = math.ceil(np.max(a)+3)

    dis_st = math.floor((point-(site_ed-site_st))/2)
                              
    label = np.zeros((n_c,point), dtype=float32)
    label[0:4, int(dis_st):int(dis_st+site_ed-site_st)] = C[:, site_st:site_ed]
    label = label/np.max(label)

    GMdata = np.zeros((point,800), dtype=float32)    
    GMdata[int(dis_st):int(dis_st+site_ed-site_st), :] = chrom[site_st:site_ed, :]
    
    return GMdata,label,S

def gauss3(loc0,spectrum,point,xa,R_range,S_range): 
    A = np.random.uniform(5e3, 8e4, (3))      
    R = np.random.uniform(R_range[0], R_range[1], (2))
    Scale = np.random.uniform(S_range[0], S_range[1], (3))

    CC = np.zeros((3,point), dtype=float32)
    for i in range(3):
        rv = norm(loc = loc0, scale = Scale[i])
        CC[i,:] = rv.pdf(xa)
             
    lw0 = loc0 - np.min(np.where((CC[0,:] > (max(CC[0,:] * 1e-3)))))
    rw0 = np.max(np.where((CC[0,:] > (max(CC[0,:] * 1e-3))))) - loc0
    lw1 = loc0 - np.min(np.where((CC[1,:] > (max(CC[1,:] * 1e-3)))))
    rw1 = np.max(np.where((CC[1,:] > (max(CC[1,:] * 1e-3))))) - loc0
    lw2 = loc0 - np.min(np.where((CC[2,:] > (max(CC[2,:] * 1e-3)))))
    rw2 = np.max(np.where((CC[2,:] > (max(CC[2,:] * 1e-3))))) - loc0

    loc1 = int(R[0]*(lw0+lw1)+loc0)
    loc2 = int(R[1]*(lw1+lw2)+loc1)
    loc = [loc0, loc1, loc2]
    
    C = np.zeros((3,point), dtype=float32)
    for i in range(3):
        rv = norm(loc = loc[i], scale = Scale[i])
        C[i,:] = rv.pdf(xa) * A[i] 

    index = random.sample(range(0,spectrum.shape[0]),3)
    S = (spectrum[index,:])
    chrom = np.dot(C.T, S)
    
    chrom = gauss_noise(chrom, np.random.randint(3,30))
    chrom = add_baseline(chrom, point)
    chrom = signal.detrend(chrom)
    chrom[chrom<0] = 0
    chrom = chrom/np.max(chrom)
    
    are = np.zeros((3,point), dtype=float32)
    ta = np.zeros((1,point), dtype=float32)
    are[0, math.floor(loc0-lw0):math.ceil(loc0+rw0)] = 1
    are[1, math.floor(loc1-lw1):math.ceil(loc1+rw1)] = 1
    are[2, math.floor(loc2-lw2):math.ceil(loc2+rw2)] = 1
    for i in range(point):
        ta[0,i] = are[0,i] + are[1,i] + are[2,i]
        if ta[0,i] >= 1:
            ta[0,i] = 1
    a = np.where(ta[0]==1)
    site_st = math.floor(np.min(a)-3)
    site_ed = math.ceil(np.max(a)+3)

    dis_st = math.floor((point-(site_ed-site_st))/2)
                              
    label = np.zeros((n_c,point), dtype=float32)
    label[0:3, int(dis_st):int(dis_st+site_ed-site_st)] = C[:, site_st:site_ed]
    label = label/np.max(label)

    GMdata = np.zeros((point,800), dtype=float32)    
    GMdata[int(dis_st):int(dis_st+site_ed-site_st), :] = chrom[site_st:site_ed, :]

    return GMdata,label,S

def gauss2(loc0,spectrum,point,xa,R_range,S_range): 
    A = np.random.uniform(5e3, 8e4, (3))      
    R = np.random.uniform(R_range[0], R_range[1], (1))
    Scale = np.random.uniform(S_range[0], S_range[1], (2))

    CC = np.zeros((2,point), dtype=float32)
    for i in range(2):
        rv = norm(loc = loc0, scale = Scale[i])
        CC[i,:] = rv.pdf(xa)

    lw0 = loc0 - np.min(np.where((CC[0,:] > (max(CC[0,:] * 1e-3)))))
    rw0 = np.max(np.where((CC[0,:] > (max(CC[0,:] * 1e-3))))) - loc0
    lw1 = loc0 - np.min(np.where((CC[1,:] > (max(CC[1,:] * 1e-3)))))
    rw1 = np.max(np.where((CC[1,:] > (max(CC[1,:] * 1e-3))))) - loc0

    loc1 = int(R[0]*(lw0+lw1)+loc0)
    loc = [loc0, loc1]
    
    C = np.zeros((2,point), dtype=float32)
    for i in range(2):
        rv = norm(loc = loc[i], scale = Scale[i])
        C[i,:] = rv.pdf(xa) * A[i] 

    index = random.sample(range(0,spectrum.shape[0]),2)
    S = (spectrum[index,:])
    chrom = np.dot(C.T, S)
    
    chrom = gauss_noise(chrom, np.random.randint(3,30))
    chrom = add_baseline(chrom, point)
    chrom = signal.detrend(chrom)
    chrom[chrom<0] = 0
    chrom = chrom/np.max(chrom)
    
    are = np.zeros((2,point), dtype=float32)
    ta = np.zeros((1,point), dtype=float32)
    are[0, math.floor(loc0-lw0):math.ceil(loc0+rw0)] = 1
    are[1, math.floor(loc1-lw1):math.ceil(loc1+rw1)] = 1
    for i in range(point):
        ta[0,i] = are[0,i] + are[1,i]
        if ta[0,i] >= 1:
            ta[0,i] = 1
    a = np.where(ta[0]==1)
    site_st = math.floor(np.min(a)-3)
    site_ed = math.ceil(np.max(a)+3)

    dis_st = math.floor((point-(site_ed-site_st))/2)
                              
    label = np.zeros((n_c,point), dtype=float32)
    label[0:2, int(dis_st):int(dis_st+site_ed-site_st)] = C[:, site_st:site_ed]
    label = label/np.max(label)

    GMdata = np.zeros((point,800), dtype=float32)    
    GMdata[int(dis_st):int(dis_st+site_ed-site_st), :] = chrom[site_st:site_ed, :]

    return GMdata,label,S

def gauss1(loc0,spectrum,point,xa,S_range): 
    A = np.random.uniform(5e3, 8e4, (3))      
    Scale = np.random.uniform(S_range[0], S_range[1], (1))

    C = np.zeros((1,point), dtype=float32)
    for i in range(1):
        rv = norm(loc = loc0, scale = Scale[i])
        C[i,:] = rv.pdf(xa) * A[i]

    lw0 = loc0 - np.min(np.where((C[0,:] > (max(C[0,:] * 1e-3)))))
    rw0 = np.max(np.where((C[0,:] > (max(C[0,:] * 1e-3))))) - loc0

    index = random.sample(range(0,spectrum.shape[0]),1)
    S = (spectrum[index,:])
    chrom = np.dot(C.T, S)
        
    chrom = gauss_noise(chrom, np.random.randint(3,30))
    chrom = add_baseline(chrom, point)
    chrom = signal.detrend(chrom)
    chrom[chrom<0] = 0
    chrom = chrom/np.max(chrom)
    
    are = np.zeros((1,point), dtype=float32)
    are[0, math.floor(loc0-lw0):math.ceil(loc0+rw0)] = 1

    a = np.where(are[0]==1)
    site_st = math.floor(np.min(a)-3)
    site_ed = math.ceil(np.max(a)+3)

    dis_st = math.floor((point-(site_ed-site_st))/2)
                              
    label = np.zeros((n_c,point), dtype=float32)
    label[0:1, int(dis_st):int(dis_st+site_ed-site_st)] = C[:, site_st:site_ed]
    label = label/np.max(label)

    GMdata = np.zeros((point,800), dtype=float32)    
    GMdata[int(dis_st):int(dis_st+site_ed-site_st), :] = chrom[site_st:site_ed, :]

    return GMdata,label,S

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

if __name__ == '__main__':
   
    savefile = 'C:/DeepEER2/data/augdata/'
    trainsave = savefile + 'train/'
    figuresave = savefile + 'train_figure/'
    validationsave = savefile + 'validation/'
    testsave = savefile + 'test/'
    
    mkdir(trainsave)
    mkdir(figuresave)
    mkdir(validationsave)
    mkdir(testsave)
    
    datafile = u'C:/Users/ZNDX001/Documents/Python_Scripts/deepEER/mass_re.npy' 
    spectrum = np.load(datafile)
    spectrum = process(spectrum)
    
    
###training datas###
    num1 = 80000
    batch_num = 1    
    n_c = 5
    point = 128    
    xa = np.arange(0, point, 1)    
    
    fn = 1
    savename = 'C:/DeepEER2/data/val_test1/0418/' + str(fn)
    mkdir(savename)
    
    loc0 = 30
    data,label,S = gauss5(loc0,spectrum,point,xa,[0.2,0.31],[3,5])

    GCMSdata=np.zeros((batch_num, point, spectrum.shape[1]))
    labeldata=np.zeros((batch_num, point, n_c))
         
    plt.figure()
    plt.subplot(211)
    plt.plot(data)
    plt.subplot(212)
    plt.plot(label.T)
    plt.savefig(savename +'/' + 'test.png')
    
    GCMSdata[0,:,:] = data
    labeldata[0,:,:] = label.T
    
    np.save(savename +'/' + 'data'+str(0)+'.npy',GCMSdata)
    np.save(savename +'/' + 'label'+str(0)+'.npy',labeldata)

    matplotlib.use('Agg')
    for n in range(0,num1):           
        GCMSdata=np.zeros((batch_num, point, spectrum.shape[1]))
        labeldata=np.zeros((batch_num, point, n_c))
        for b in range(batch_num):
            COM = np.random.randint(1,101)     
            if COM in range(1,21):
                loc0 = 63
                data,label,S = gauss1(loc0,spectrum,point,xa, [2,9])
            if COM in range(21,26):
                loc0 = 30
                data,label,S = gauss2(loc0,spectrum,point,xa, [0.2,0.4], [3,5])
            if COM in range(26,41):
                loc0 = 30
                data,label,S = gauss2(loc0,spectrum,point,xa, [0.4,0.8], [2,5])
            if COM in range(41,46):
                loc0 = 30
                data,label,S = gauss3(loc0,spectrum,point,xa, [0.2,0.4], [3,5])
            if COM in range(46,61):
                loc0 = 30
                data,label,S = gauss3(loc0,spectrum,point,xa, [0.3,0.8], [3,5])
            if COM in range(61,66):
                loc0 = 30
                data,label,S = gauss4(loc0,spectrum,point,xa, [0.2, 0.4], [3,5])
            if COM in range(66,81):
                loc0 = 30
                data,label,S = gauss4(loc0,spectrum,point,xa, [0.3, 0.7], [3,5])
            if COM in range(81,86):
                loc0 = 30
                data,label,S = gauss5(loc0,spectrum,point,xa, [0.2, 0.4], [3,4])
            if COM in range(86,101):
                loc0 = 30
                data,label,S = gauss5(loc0,spectrum,point,xa, [0.3, 0.6], [3,4])
                                         
            GCMSdata [b,:,:] = data
            labeldata [b,:,:] = label.T
            
        if n%400 == 0:
            plt.figure()
            plt.subplot(211)
            plt.plot(data)
            plt.subplot(212)
            plt.plot(label.T)
            plt.savefig(figuresave + 'f'+str(n)+'.png')

        np.save(trainsave + 'data'+str(n)+'.npy',GCMSdata)
        np.save(trainsave + 'label'+str(n)+'.npy',labeldata)
        if n%100==0:
            print(n_c,int(n*batch_num),'samples finished')
            
###validation datas###
    num1 = 10000
    batch_num = 1    
    n_c = 5
    point = 128    
    xa = np.arange(0, point, 1)    

    for n in range(0,num1):           
        GCMSdata=np.zeros((batch_num, point, spectrum.shape[1]))
        labeldata=np.zeros((batch_num, point, n_c))
        for b in range(batch_num):
            COM = np.random.randint(1,101)     
            if COM in range(1,21):
                loc0 = 63
                data,label,S = gauss1(loc0,spectrum,point,xa, [2,9])
            if COM in range(21,26):
                loc0 = 30
                data,label,S = gauss2(loc0,spectrum,point,xa, [0.2,0.4], [3,5])
            if COM in range(26,41):
                loc0 = 30
                data,label,S = gauss2(loc0,spectrum,point,xa, [0.4,0.8], [2,5])
            if COM in range(41,46):
                loc0 = 30
                data,label,S = gauss3(loc0,spectrum,point,xa, [0.2,0.4], [3,5])
            if COM in range(46,61):
                loc0 = 30
                data,label,S = gauss3(loc0,spectrum,point,xa, [0.3,0.8], [3,5])
            if COM in range(61,66):
                loc0 = 30
                data,label,S = gauss4(loc0,spectrum,point,xa, [0.2, 0.4], [3,5])
            if COM in range(66,81):
                loc0 = 30
                data,label,S = gauss4(loc0,spectrum,point,xa, [0.3, 0.7], [3,5])
            if COM in range(81,86):
                loc0 = 30
                data,label,S = gauss5(loc0,spectrum,point,xa, [0.2, 0.4], [3,4])
            if COM in range(86,101):
                loc0 = 30
                data,label,S = gauss5(loc0,spectrum,point,xa, [0.3, 0.6], [3,4])
                                         
            GCMSdata [b,:,:] = data
            labeldata [b,:,:] = label.T

        np.save(validationsave + 'data'+str(n)+'.npy',GCMSdata)
        np.save(validationsave + 'label'+str(n)+'.npy',labeldata)
        if n%100==0:
            print(n_c,int(n*batch_num),'samples finished')
            
###test datas###
    num1 = 10000
    batch_num = 1    
    n_c = 8
    point = 256    
    xa = np.arange(0, point, 1)    

    for n in range(0,num1):           
        GCMSdata=np.zeros((batch_num, point, spectrum.shape[1]))
        labeldata=np.zeros((batch_num, point, n_c))
        for b in range(batch_num):
            COM = np.random.randint(1,101)     
            if COM in range(1,21):
                loc0 = 63
                data,label,S = gauss1(loc0,spectrum,point,xa, [2,9])
            if COM in range(21,26):
                loc0 = 30
                data,label,S = gauss2(loc0,spectrum,point,xa, [0.2,0.4], [3,5])
            if COM in range(26,41):
                loc0 = 30
                data,label,S = gauss2(loc0,spectrum,point,xa, [0.4,0.8], [2,5])
            if COM in range(41,46):
                loc0 = 30
                data,label,S = gauss3(loc0,spectrum,point,xa, [0.2,0.4], [3,5])
            if COM in range(46,61):
                loc0 = 30
                data,label,S = gauss3(loc0,spectrum,point,xa, [0.3,0.8], [3,5])
            if COM in range(61,66):
                loc0 = 30
                data,label,S = gauss4(loc0,spectrum,point,xa, [0.2, 0.4], [3,5])
            if COM in range(66,81):
                loc0 = 30
                data,label,S = gauss4(loc0,spectrum,point,xa, [0.3, 0.7], [3,5])
            if COM in range(81,86):
                loc0 = 30
                data,label,S = gauss5(loc0,spectrum,point,xa, [0.2, 0.4], [3,4])
            if COM in range(86,101):
                loc0 = 30
                data,label,S = gauss5(loc0,spectrum,point,xa, [0.3, 0.6], [3,4])
                                         
            GCMSdata [b,:,:] = data
            labeldata [b,:,:] = label.T

        np.save(testsave + 'data'+str(n)+'.npy',GCMSdata)
        np.save(testsave + 'label'+str(n)+'.npy',labeldata)
        if n%100==0:
            print(n_c,int(n*batch_num),'samples finished')















