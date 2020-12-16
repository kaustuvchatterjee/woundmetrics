#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:36:15 2020

@author: kaustuv
"""

import streamlit as st
from streamlit import caching
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
#from skimage import io
#import cv2

from test3 import woundseg

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

st.title("Wound Metrics")
"""

"""
st.markdown("### Load Image")
img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg'])
caching.clear_cache()

if img_file is not None:
    bytes_data = img_file.read()
    img = load_image(img_file)
#    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
#    I = cv2.imdecode(file_bytes, 1)
    # Resize Image
    w,h = img.size
    r = h/w
    
    if w != 256:
        w = 256
        h = int(w*r)
    img = img.resize((w,h))
    
#    st.image(I)
    
    #Calibrate Image
    st.markdown("""To calibrate image place top left and right edges of the box over two points of known distance in millimeters.""")
    box = st_cropper(img, realtime_update=True, 
                     aspect_ratio = (1,0.01),
                     box_color="green",
                     return_type='box', key = 1)
    dist = st.number_input("Distance in mm:", value=30)
    w = box["width"]
    res = dist/w
    st.write("Image Resolution = ", round(res,2),"  mm/pixel")
         
#    # ROI
    st.markdown("### Select ROI")
    box = st_cropper(img, realtime_update=True, 
             aspect_ratio = None,
             box_color="green",
             return_type='box', key = 2)

    roi = np.zeros(4)
    roi[0] = box["left"]
    roi[1] = box["left"]+box["width"]-1
    roi[2] = box["top"]
    roi[3] = box["top"]+box["height"]-1

#    #Segment Image
    if bytes_data is not None:
        fig, area, perim, laxis, saxis, gI, sI, nI, gP, sP, nP,t1, t2, t3 = woundseg(bytes_data,roi, res)

    st.markdown("### Segmented Wound")
    st.image(fig)

    st.write("Area: ", area, "sqmm")
    st.write("Perimeter: ", perim, "mm")
    st.write("Long Axis: ", laxis, "mm")
    st.write("Short Axis: ", saxis, "mm")
    
    gText = "Granulation Tissue ("+str(gP)+"%)"
    sText = "Slough ("+str(sP)+"%)"
    nText = "Necrotic Tissue ("+str(nP)+"%)"
    
    st.image(gI,caption=gText)
    st.image(sI,caption=sText)
    st.image(nI,caption=nText)

# Clear cache and re-run

st.button('Rerun')

