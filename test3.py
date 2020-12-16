#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:44:53 2020

@author: kaustuv
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage import morphology
import cv2
import streamlit as st
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
#from skimage.transform import resize

def woundseg(bytes_data,roi, res):

    # Image manipulation

    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    I = cv2.imdecode(file_bytes, 1)

    h = I.shape[0]
    w = I.shape[1]
    r = h/w
    
    if w != 256:
        w = 256
        h = int(w*r)
    I = cv2.resize(I,(w,h))
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    Ig = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    Il = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(Il)
    
    x1 = int(roi[0])
    x2 = int(roi[1])
    y1 = int(roi[2])
    y2 = int(roi[3])
    
    mask = np.zeros((Ig.shape), dtype=bool)
    mask[y1:y2,x1:x2]=True
    
    
    # Edge detection
    low_threshold = st.slider("Threshold:",1,100,30)
    ratio = st.slider("Width:",2.0,4.0,3.0)
#    ratio = 3
    kernel_size = 3

#    low_threshold = 30
    Ib = cv2.blur(Ig, (3,3))
    
    Ie = cv2.Canny(Ib, low_threshold, low_threshold*ratio, kernel_size)
    Ie[mask == 0]=0
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bw = cv2.morphologyEx(Ie, cv2.MORPH_CLOSE, kernel,iterations=6)
    bw1 = ndi.binary_fill_holes(bw)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bw2 = cv2.erode(np.float32(bw1),se,iterations = 3).astype(bool)
    
    
    #label segmented regions
    label_img = label(bw2)
    regions = regionprops(label_img)
    
    
    areas=[]
    
    for i in range(len(regions)):
        areas.append(regions[i].area)
    
    max_area = np.max(areas)
    #    print(max_area)
    bw2[bw2==255]=1
    bw2 = bw2.astype(bool)
    
    bw3 = morphology.remove_small_objects(bw2,max_area-1)
    
    bw3 = np.uint8(bw3)
    bw3[bw3==1]=255
    
    f = np.uint8(I.copy())
    img = bw3
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    a, b = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = cv2.drawContours(f, a, 0, (0,255,0), 2)

    #------------------------
    # Morphological calculations
    #------------------------
    props = regions[0]
    
    pArea = props["area"]
    pPerimeter = props["perimeter"]
    lAxis = props["major_axis_length"]
    sAxis = props["minor_axis_length"]

    wArea = round(pArea*res**2,2)
    wPerimeter = round(pPerimeter*res,2)
    lAxis = round(lAxis*res,2)
    sAxis = round(sAxis*res,2)
    
    
    #--------------------------------
    #Tissue Typing
    #--------------------------------
    fI = np.uint8(Il.copy())
    Il = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(Il)
    
    fg = cv2.bitwise_and(fI, fI, mask=bw3)
    Is = cv2.bitwise_and(I, I, mask=bw3)
    fgc = fg[:,:,[1,2]]
    im_reshaped = fgc.reshape(fgc.shape[0]*fgc.shape[1], fgc.shape[2])
    im_reshaped = np.float32(im_reshaped)
#    im_reshaped[im_reshaped==0]=float("NaN")
#    im_reshaped = fg.reshape((-1, 3))
    im_reshaped = np.float32(im_reshaped)
    # define criteria and apply kmeans()
    k = 4
    ref = np.array([[165,160],[135,150],[120,120],[0,0]])
#    ref = np.array([[165,160],[135,150],[127,127],[0,0]])
#    kmeans = KMeans(n_clusters=k, init=seed).fit(im_reshaped)
#    centers = kmeans.cluster_centers_
#    labels = kmeans.labels_
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,labels,centers=cv2.kmeans(im_reshaped,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)
#    
#    # flatten the labels array
    labels = labels.flatten()
#    st.write("Seed: ",seed,"centers: ", centers, "Labels: ", labels)

    dist = np.zeros(k)
    tissue_class = np.zeros(k)
    
    for i in range(k):
        for j in range(k):
            dist[j-1] = np.linalg.norm(centers[i-1]-ref[j-1])
        tissue_class[i-1] = np.argmin(dist)

            
    g = np.zeros(fg.shape,dtype='uint8')
    s = np.zeros(fg.shape,dtype='uint8')
    n = np.zeros(fg.shape,dtype='uint8')
    
    g = g.reshape((-1,3))
    s = s.reshape((-1,3))
    n = n.reshape((-1,3))
    
    for i in range(k):
        if tissue_class[i] == 0:
            g[labels==i] = [1,1,1]
            
        if tissue_class[i] == 1:
            s[labels==i] = [1,1,1]
 
        if tissue_class[i] == 2:
            n[labels==i] = [1,1,1]
            
    
    g = g.reshape(fg.shape)
    s = s.reshape(fg.shape)
    n = n.reshape(fg.shape)
    
    gPixels = g.sum()
    sPixels = s.sum()
    nPixels = n.sum()
    
    totP = gPixels+sPixels+nPixels
    gP = round(gPixels*100/totP,1)
    sP = round(sPixels*100/totP,1)
    nP = round(nPixels*100/totP,1)
    
    gI = cv2.bitwise_and(Is, I, mask=g[:,:,1])
    sI = cv2.bitwise_and(Is, I, mask=s[:,:,1])
    nI = cv2.bitwise_and(Is, I, mask=n[:,:,1])
 
    bg = np.zeros(fg.shape,dtype='uint8')
    bg[:,:] = (2,89,15)

    bw4 = cv2.cvtColor(gI, cv2.COLOR_RGB2GRAY)
    bw4[bw4!=0]=255
    bw4 = cv2.bitwise_not(bw4)
    bgt = cv2.bitwise_and(bg, bg, mask=bw4)
    gI = cv2.add(gI,bgt)
 
    bw4 = cv2.cvtColor(sI, cv2.COLOR_RGB2GRAY)
    bw4[bw4!=0]=255
    bw4 = cv2.bitwise_not(bw4)
    bgt = cv2.bitwise_and(bg, bg, mask=bw4)
    sI = cv2.add(sI,bgt)    
 
    
    bw4 = cv2.cvtColor(nI, cv2.COLOR_RGB2GRAY)
    bw4[bw4!=0]=255
    bw4 = cv2.bitwise_not(bw4)
    bgt = cv2.bitwise_and(bg, bg, mask=bw4)
    nI = cv2.add(nI,bgt)
    #Cluster 0 Mask
    mask0 = np.zeros(fg.shape,dtype='uint8')
    mask0 = mask0.reshape((-1,3))
    cluster = 0
    mask0[labels==cluster] = [1,1,1]
    mask0 = mask0.reshape(fg.shape)
 
    
    #Cluster 1 Mask
    mask1 = np.zeros(fg.shape,dtype='uint8')
    mask1 = mask1.reshape((-1,3))
    cluster = 1
    mask1[labels==cluster] = [1,1,1]
    mask1 = mask1.reshape(fg.shape)

    #Cluster 2 Mask
    mask2 = np.zeros(fg.shape,dtype='uint8')
    mask2 = mask2.reshape((-1,3))
    cluster = 2
    mask2[labels==cluster] = [1,1,1]
    mask2 = mask2.reshape(fg.shape)

#    #Cluster 3 Mask
#    mask3 = np.zeros(fg.shape,dtype='uint8')
#    mask3 = mask3.reshape((-1,3))
#    cluster = 3
#    mask3[labels==cluster] = [1,1,1]
#    mask3 = mask3.reshape(fg.shape)
    
    # Apply Masks
    t1 = cv2.bitwise_and(Is, I, mask=mask0[:,:,1])
    t2 = cv2.bitwise_and(Is, I, mask=mask1[:,:,1])
    t3 = cv2.bitwise_and(Is, I, mask=mask2[:,:,1])
#    t4 = cv2.bitwise_and(Is, I, mask=mask3[:,:,1])
    

    return contour, wArea, wPerimeter, lAxis, sAxis, gI, sI, nI, gP, sP, nP, t1, t2, t3
